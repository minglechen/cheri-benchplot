#include <iostream>
#include <cassert>
#include <cstdint>
#include <iterator>
#include <mutex>
#include <optional>
#include <string_view>

#include <llvm/BinaryFormat/Dwarf.h>
#include <llvm/DebugInfo/DIContext.h>
#include <llvm/DebugInfo/DWARF/DWARFContext.h>
#include <llvm/MC/TargetRegistry.h>
#include <llvm/Object/Binary.h>
#include <llvm/Object/ObjectFile.h>
#include <llvm/Support/Error.h>
#include <llvm/Support/Format.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Target/TargetOptions.h>
#include <llvm/Target/TargetMachine.h>

#include "dwarf.h"

namespace object = llvm::object;
namespace dwarf = llvm::dwarf;

using FileLineInfoKind = llvm::DILineInfoSpecifier::FileLineInfoKind;
using InfoOrError = llvm::Expected<std::optional<cheri_benchplot::TypeInfo>>;

namespace {

using namespace cheri_benchplot;

std::once_flag LLVMInitFlag;

/**
 * Helper to build errors
 */
llvm::Error makeError(std::string_view msg) {
  return llvm::make_error<llvm::StringError>(msg.data(),
                                             llvm::inconvertibleErrorCode());
}

/**
 * Helper for string formatting
 */
template <typename... TArgs>
inline std::string format(const char *Fmt, const TArgs &...Args) {
  std::size_t Len = std::snprintf(nullptr, 0, Fmt, Args...) + 1;
  std::unique_ptr<char[]> Data(new char[Len]);
  std::snprintf(Data.get(), Len, Fmt, Args...);
  return std::string(Data.get(), Data.get() + Len - 1);
}

/**
 * Extract name attribute from DIE.
 */
llvm::Optional<std::string> extractName(const llvm::DWARFDie &Die) {
  if (auto Name = dwarf::toString(Die.find(dwarf::DW_AT_name))) {
    if (!Name)
      return llvm::None;
    return std::string(*Name);
  }
  return llvm::None;
}

/*
 * Extract name attribute from DIE or build an anonymous identifier from
 * the source file/line.
 */
std::string extractNameOrAnon(const llvm::DWARFDie &Die) {
  auto MaybeName = extractName(Die);
  if (MaybeName)
    return *MaybeName;
  std::string File = Die.getDeclFile(FileLineInfoKind::AbsoluteFilePath);
  unsigned long Line = Die.getDeclLine();
  return format("<anon>.%s.%ld", File.c_str(), Line);
}

/**
 * Base class to visit the DIE tree.
 */
class DieVisitorBase {
public:
  DieVisitorBase(llvm::DWARFContext &DICtx) : DICtx{DICtx} {}
  virtual ~DieVisitorBase() = default;

  virtual llvm::Expected<bool> beginUnit(llvm::DWARFDie &CUDie);

#define HANDLE_DW_TAG(ID, NAME, VERSION, VENDOR, KIND)                         \
  virtual llvm::Expected<bool> visit_##NAME(llvm::DWARFDie &Die) {             \
    return false;                                                              \
  }
#include <llvm/BinaryFormat/Dwarf.def>
#undef HANDLE_DW_TAG

protected:
  llvm::DWARFContext &DICtx;
  llvm::DWARFUnit *CurrentUnit;
  std::string CurrentUnitName;
};

llvm::Expected<bool> DieVisitorBase::beginUnit(llvm::DWARFDie &CUDie) {
  // Check for validity
  if (!CUDie) {
    return makeError("Invalid Compile Unit DIE");
  }

  auto AtName = CUDie.find(dwarf::DW_AT_name);
  CurrentUnitName = "<UKNOWN>";
  if (AtName) {
    llvm::Expected NameOrErr = AtName->getAsCString();
    if (NameOrErr)
      CurrentUnitName = *NameOrErr;
  }
  return true;
}

template <typename V>
llvm::Expected<bool> visitDie(V &Visitor, llvm::DWARFDie &Die) {
#define HANDLE_DW_TAG(ID, NAME, VERSION, VENDOR, KIND)                         \
  case ID:                                                                     \
    return Visitor.visit_##NAME(Die);

  switch (Die.getTag()) {
#include <llvm/BinaryFormat/Dwarf.def>
  default:
    // Make this an error?
    return true;
  }
#undef HANDLE_DW_TAG
}

class TypeLayoutVisitor : public DieVisitorBase, public TypeInfoContainer {
public:
  using TypeInfoPtr = std::shared_ptr<TypeInfo>;

  TypeLayoutVisitor(llvm::DWARFContext &DICtx) : DieVisitorBase(DICtx), TypeInfoContainer() {
    const llvm::DWARFObject &DWObj = DICtx.getDWARFObj();
    ArchIsLittleEndian = DWObj.isLittleEndian();

    // const object::ObjectFile *Obj = DWObj.getFile();
    // llvm::Triple TheTriple = Obj->makeTriple();
    // std::string Error;
    // const llvm::Target *TheTarget = llvm::TargetRegistry::lookupTarget(TheTriple.getArchName().str(), TheTriple, Error);
    // if (!TheTarget)
    //   llvm::errs() << Error << "\n";
    // llvm::SubtargetFeatures Features = Obj->getFeatures();
    // std::string TripleName = TheTriple.getTriple();
    // std::string MCPU = Obj->tryGetCPUName().getValueOr("").str();
    // llvm::TargetOptions Options;
    // std::unique_ptr<const llvm::TargetMachine> TM(TheTarget->createTargetMachine(
    //     TripleName, MCPU, Features.getString(), Options, llvm::None));

    // llvm::outs() << "Triple: " << TripleName << " Features: " << Features.getString() <<
    //     " PtrSize=" << TM->getPointerSize(0) << "\n";
    ArchPointerSize = 8;
  }

  virtual llvm::Expected<bool>
  visit_structure_type(llvm::DWARFDie &Die) override;
  virtual llvm::Expected<bool> visit_class_type(llvm::DWARFDie &Die) override;
  virtual llvm::Expected<bool> visit_union_type(llvm::DWARFDie &Die) override;
  virtual llvm::Expected<bool> visit_typedef(llvm::DWARFDie &Die) override;

  virtual std::optional<TypeInfo> findComposite(unsigned long Handle) const override;
  virtual Iterator beginComposite() const override;
  virtual Iterator endComposite() const override;

private:
  llvm::Expected<bool> visitComposite(const llvm::DWARFDie &Die);
  llvm::Expected<TypeInfoPtr> makeTypeInfo(const llvm::DWARFDie &Die,
                                           bool Nested);
  llvm::Error makeCompositeTypeInfo(const char *Prefix,
                                    const llvm::DWARFDie &Die,
                                    bool Nested,
                                    TypeInfoPtr TI);
  llvm::Error makeSubroutineTypeInfo(const llvm::DWARFDie &Die,
                                     TypeInfoPtr TI);

  bool ArchIsLittleEndian;
  unsigned int ArchPointerSize;
  // Collect type information for composite types
  std::unordered_map<unsigned long, TypeInfoPtr> CompositeTypeInfo; 
  // Collect all type information by DIE offset to avoid duplication
  std::unordered_map<unsigned long, TypeInfoPtr> TypeInfoByOffset;
};

/*
 * Build type info from a Die subtree defining a type as referenced by
 * DW_AT_type
 */
llvm::Expected<TypeLayoutVisitor::TypeInfoPtr>
TypeLayoutVisitor::makeTypeInfo(const llvm::DWARFDie &Die, bool Nested) {
  // Try to find a cached typeinfo
  auto Pos = TypeInfoByOffset.find(Die.getOffset());
  if (Pos != TypeInfoByOffset.end()) {
    return Pos->second;
  }
  // We don't have one cached, so we have to make a new one
  auto TI = std::make_shared<TypeInfo>();
  TI->Handle = Die.getOffset();
  // Prevent re-entering for the same Die
  TypeInfoByOffset[TI->Handle] = TI;

  std::vector<llvm::DWARFDie> Chain;
  llvm::DWARFDie Next = Die;
  while (Next) {
    Chain.push_back(Next);
    if (Next.getTag() == dwarf::DW_TAG_structure_type ||
        Next.getTag() == dwarf::DW_TAG_class_type ||
        Next.getTag() == dwarf::DW_TAG_union_type ||
        Next.getTag() == dwarf::DW_TAG_subroutine_type ||
        Next.getTag() == dwarf::DW_TAG_enumeration_type ||
        Next.getTag() == dwarf::DW_TAG_base_type) {
      break;
    }
    Next = Next.getAttributeValueAsReferencedDie(dwarf::DW_AT_type)
               .resolveTypeUnitReference();
  }
  // Handle void if the chain does not break with one of the tags above
  if (!Next) {
    TI->BaseName = "void";
    TI->TypeName = "void";
  }

  // Evaluate the type definition
  if (Chain.size() == 0)
    return makeError("Could not resolve type");

  for (auto IterDie = Chain.rbegin(); IterDie != Chain.rend(); ++IterDie) {
    llvm::DWARFDie &D = *IterDie;

    switch (D.getTag()) {
    case dwarf::DW_TAG_base_type: {
      auto Name = extractName(D);
      if (!Name)
        return makeError("Base type without a name");
      auto Size = dwarf::toUnsigned(D.find(dwarf::DW_AT_byte_size));
      if (!Size)
        return makeError("Base type without a size");
      TI->BaseName = *Name;
      TI->TypeName = *Name;
      TI->Size = *Size;
      break;
    }
    case dwarf::DW_TAG_pointer_type: {
      // XXX must fetch arch pointer size somehow
      TI->Size = dwarf::toUnsigned(D.find(dwarf::DW_AT_byte_size),
                                   ArchPointerSize);
      TI->TypeName += " *";
      // Note, reset the flags because this is now a pointer
      TI->Flags = kIsPtr;
      break;
    }
    case dwarf::DW_TAG_const_type: {
      TI->TypeName += " const";
      TI->Flags |= kIsConst;
      break;
    }
    case dwarf::DW_TAG_volatile_type: {
      TI->TypeName += " volatile";
      break;
    }
    case dwarf::DW_TAG_array_type: {
      llvm::DWARFDie SubrangeDie;
      for (auto &Child : D) {
        if (Child.getTag() == dwarf::DW_TAG_subrange_type) {
          SubrangeDie = Child;
          break;
        }
      }

      if (!SubrangeDie) {
        return makeError("Array type without subrange");
      }
      auto Count = SubrangeDie.find(dwarf::DW_AT_count);
      auto UpperBound = SubrangeDie.find(dwarf::DW_AT_upper_bound);
      unsigned long NItems = 0;
      if (Count) {
        auto UnsignedCount = dwarf::toUnsigned(*Count);
        if (!UnsignedCount) {
          return makeError("Unexpected item count type");
        }
        NItems = *UnsignedCount;
      } else if (UpperBound) {
        auto UnsignedCount = dwarf::toUnsigned(*UpperBound);
        if (!UnsignedCount) {
          return makeError("Unexpected array upper bound type");
        }
        NItems = *UnsignedCount + 1;
      }
      TI->TypeName += format(" [%ld]", NItems);
      TI->ArrayItems = NItems;
      TI->Size = TI->Size * NItems;
      TI->Flags |= kIsArray;
      break;
    }
    case dwarf::DW_TAG_subroutine_type: {
      if (auto E = makeSubroutineTypeInfo(D, TI)) {
        return E;
      }
      break;
    }
    case dwarf::DW_TAG_typedef: {
      auto Name = extractName(D);
      if (!Name)
        return makeError("Typedef without a name");
      TI->TypeName = *Name;
      break;
    }
    case dwarf::DW_TAG_structure_type:
    case dwarf::DW_TAG_class_type: {
      if (auto E = makeCompositeTypeInfo("struct ", D, Nested, TI)) {
        return E;
      }
      TI->Flags |= kIsStruct;
      break;
    }
    case dwarf::DW_TAG_union_type: {
      if (auto E = makeCompositeTypeInfo("union ", D, Nested, TI)) {
        return E;
      }
      TI->Flags |= kIsUnion;
      break;
    }
    case dwarf::DW_TAG_enumeration_type: {
      auto Name = extractNameOrAnon(D);
      TI->TypeName = "enum " + Name;
      TI->BaseName = "enum " + Name;
      if (Die.find(dwarf::DW_AT_declaration))
        continue;
      auto Size = dwarf::toUnsigned(D.find(dwarf::DW_AT_byte_size));
      if (!Size)
        return makeError("Enum type without a size");
      TI->Size = *Size;
      break;
    }
    default:
      return makeError(format("Type resolver did not handle %s",
                              dwarf::TagString(D.getTag()).str().c_str()));
    }
  }

  return TI;
}

llvm::Error
TypeLayoutVisitor::makeCompositeTypeInfo(const char *Prefix,
                                         const llvm::DWARFDie &Die,
                                         bool Nested,
                                         TypeInfoPtr TI) {
  // This is a forward declaration, mark it as such.
  // Do not generate anything else because it will not be used. We could
  // resolve forward declarations after all typeinfo are collected if we needed.
  if (Die.find(dwarf::DW_AT_declaration)) {
    TI->BaseName = extractNameOrAnon(Die);
    TI->TypeName = Prefix + TI->BaseName;
    TI->Flags |= kIsDecl;
    return llvm::Error::success();
  }

  auto MaybeSize = dwarf::toUnsigned(Die.find(dwarf::DW_AT_byte_size));
  if (!MaybeSize) {
    return makeError(format("Missing struct size at %lx", Die.getOffset()));
  }
  TI->Size = *MaybeSize;
  TI->File = Die.getDeclFile(FileLineInfoKind::AbsoluteFilePath);
  TI->Line = Die.getDeclLine();

  TI->BaseName = extractNameOrAnon(Die);
  TI->TypeName = Prefix + TI->BaseName;
  if (!Die.find(dwarf::DW_AT_name).hasValue()) {
    TI->Flags |= kIsAnonymous;
  }

  // Fail if we find a specification, we need to handle this case with
  // findRecursively()
  if (Die.find(dwarf::DW_AT_specification)) {
    return makeError("DW_AT_specification unsupported");
  }

  // Now extract the structure layout
  for (auto &Child : Die.children()) {
    if (Child.getTag() == dwarf::DW_TAG_member) {
      unsigned long Offset =
          dwarf::toUnsigned(Child.find(dwarf::DW_AT_data_member_location), 0);
      unsigned long BitOffset =
          dwarf::toUnsigned(Child.find(dwarf::DW_AT_bit_offset), 0);
      std::string DefaultName = format("<anon>.%ld", Offset);
      if (BitOffset) {
        DefaultName += format(":%ld", BitOffset);
      }
      Member MI{
          .Name = dwarf::toString(Child.find(dwarf::DW_AT_name),
                                  DefaultName.c_str()),
          .Offset = Offset,
          .BitOffset = BitOffset
      };

      auto MType = Child.getAttributeValueAsReferencedDie(dwarf::DW_AT_type)
                       .resolveTypeUnitReference();
      auto MemberInfoOrErr = makeTypeInfo(MType, /*Nested=*/true);
      if (auto E = MemberInfoOrErr.takeError()) {
        return E;
      }
      MI.Type = *MemberInfoOrErr;

      auto ByteSize = Child.find(dwarf::DW_AT_byte_size);
      if (ByteSize) {
        MI.Size = dwarf::toUnsigned(ByteSize, 0);
      } else {
        MI.Size = (*MemberInfoOrErr)->Size;
      }
      auto BitSize = Child.find(dwarf::DW_AT_bit_size);
      if (BitSize) {
        MI.BitSize = dwarf::toUnsigned(BitSize, 0);
      } else {
        MI.BitSize = MI.Size * 8;
      }

      TI->Layout.push_back(MI);
    }
  }

  return llvm::Error::success();
}

llvm::Error
TypeLayoutVisitor::makeSubroutineTypeInfo(const llvm::DWARFDie &Die,
                                          TypeInfoPtr TI) {
  std::string ReturnName;
  auto ReturnType = Die.getAttributeValueAsReferencedDie(dwarf::DW_AT_type)
                        .resolveTypeUnitReference();
  if (ReturnType) {
    auto ReturnTIOrErr = makeTypeInfo(ReturnType, /*Nested=*/false);
    if (auto E = ReturnTIOrErr.takeError()) {
      return E;
    }
    ReturnName = (*ReturnTIOrErr)->TypeName;
  } else {
    ReturnName = "void";
  }

  std::vector<TypeInfoPtr> Params;
  for (auto &Child : Die.children()) {
    if (Child.getTag() != dwarf::DW_TAG_formal_parameter) {
      // XXX variadics are not supported yet, DW_TAG_unspecified_type
      continue;
    }
    auto ParamType = Child.getAttributeValueAsReferencedDie(dwarf::DW_AT_type)
                         .resolveTypeUnitReference();
    auto ParamTIOrErr = makeTypeInfo(ParamType, /*Nested=*/false);
    if (auto E = ParamTIOrErr.takeError()) {
      return E;
    }
    Params.push_back(*ParamTIOrErr);
  }

  std::string ParamString;
  for (int Idx = 0; Idx < Params.size(); ++Idx) {
    ParamString += Params[Idx]->TypeName;
    if (Idx != Params.size() - 1) {
      ParamString += ", ";
    }
  }

  std::string Name = format("%s(%s)", ReturnName.c_str(), ParamString.c_str());
  TI->TypeName = Name;
  TI->BaseName = Name;
  TI->Flags |= kIsFnPtr;

  return llvm::Error::success();
}

llvm::Expected<bool>
TypeLayoutVisitor::visitComposite(const llvm::DWARFDie &Die) {
  // Skip declarations, we don't care.
  if (Die.find(dwarf::DW_AT_declaration)) {
    return false;
  }

  auto TIOrErr = makeTypeInfo(Die, /*Nested=*/false);
  if (auto E = TIOrErr.takeError()) {
    return E;
  }
  std::shared_ptr<TypeInfo> TI = *TIOrErr;

  // Skip duplicate definitions
  auto Item = CompositeTypeInfo.find(TI->Handle);
  if (Item == CompositeTypeInfo.end()) {
    CompositeTypeInfo[TI->Handle] = TI;
  }

  return false;
}

llvm::Expected<bool>
TypeLayoutVisitor::visit_structure_type(llvm::DWARFDie &Die) {
  return visitComposite(Die);
}

llvm::Expected<bool>
TypeLayoutVisitor::visit_union_type(llvm::DWARFDie &Die) {
  return visitComposite(Die);
}

llvm::Expected<bool>
TypeLayoutVisitor::visit_class_type(llvm::DWARFDie &Die) {
  return visitComposite(Die);
}

llvm::Expected<bool> TypeLayoutVisitor::visit_typedef(llvm::DWARFDie &Die) {
  auto Name = extractName(Die);
  if (!Name)
    return makeError("typedef without a name");
  auto TypeDie = Die.getAttributeValueAsReferencedDie(dwarf::DW_AT_type)
                 .resolveTypeUnitReference();
  if (!TypeDie)
    return makeError("typedef without a type");
  auto TIOrErr = makeTypeInfo(TypeDie, /*Nested=*/false);
  if (auto E = TIOrErr.takeError())
    return E;

  /*
   * If the typedef is aliasing a structure/union name,
   * we create another entry in the CompositeTypeInfo map.
   * Otherwise, just ignore the typedef as it is not interesting here.
   */
  std::shared_ptr<TypeInfo> TI = *TIOrErr;
  TI->AliasNames.insert(*Name);

  return false;
}

std::optional<TypeInfo> TypeLayoutVisitor::findComposite(unsigned long Handle) const {
  auto Item = CompositeTypeInfo.find(Handle);
  if (Item != CompositeTypeInfo.end()) {
    return *(Item->second);
  }
  return std::nullopt;
}

TypeLayoutVisitor::Iterator TypeLayoutVisitor::beginComposite() const {
  return Iterator(CompositeTypeInfo.begin());
}

TypeLayoutVisitor::Iterator TypeLayoutVisitor::endComposite() const {
  return Iterator(CompositeTypeInfo.end());
}

} // namespace

namespace cheri_benchplot {

class DWARFHelper::HelperImpl {
public:
  HelperImpl(std::string Path);

  // int getABIPointerSize() const;
  // int getABICapabilitySize() const;
  std::unique_ptr<TypeInfoContainer> collectTypeInfo();

private:
  InfoOrError makeTypeInfo(llvm::DWARFDie &Die);

  std::string Path;
  std::unique_ptr<llvm::DWARFContext> DICtx;
  llvm::object::OwningBinary<llvm::object::Binary> OwnedBinary;
};

DWARFHelper::HelperImpl::HelperImpl(std::string Path) : Path(Path) {
  std::call_once(LLVMInitFlag, []() {
    llvm::InitializeAllTargetInfos();
    llvm::InitializeAllTargetMCs();
  });

  llvm::Expected<object::OwningBinary<object::Binary>> BinOrErr =
      object::createBinary(Path);
  if (auto E = BinOrErr.takeError()) {
    llvm::errs() << "Can not load binary " << Path << "\n";
    exit(EXIT_FAILURE);
  }
  OwnedBinary = std::move(*BinOrErr);
  auto *Obj = llvm::dyn_cast<object::ObjectFile>(OwnedBinary.getBinary());
  if (Obj == nullptr) {
    llvm::errs() << "Invalid binary at " << Path << " not an object\n";
    exit(EXIT_FAILURE);
  }

  DICtx = llvm::DWARFContext::create(
      *Obj, llvm::DWARFContext::ProcessDebugRelocations::Process, nullptr, "",
      nullptr);

  // Check DWARF version
  assert(DICtx->getMaxVersion() >= 4 &&
         "Unsupported DWARF version, use 4 or above");
}

std::unique_ptr<TypeInfoContainer> DWARFHelper::HelperImpl::collectTypeInfo() {

  auto V = std::make_unique<TypeLayoutVisitor>(*DICtx);

  for (auto &Unit : DICtx->info_section_units()) {

    dwarf::DwarfFormat DWFormat = Unit->getFormat();

    // Check if this is a compile unit
    if (isCompileUnit(Unit)) {
      assert(Unit->getVersion() >= 4 && "Unsupported DWARF version");

      llvm::DWARFDie CUDie = Unit->getUnitDIE(false);
      auto Result = V->beginUnit(CUDie);
      if (auto E = Result.takeError()) {
        llvm::errs() << "Can not visit Unit" << E << "\n";
        llvm::consumeError(std::move(E));
        continue;
      }

      // Iterate over DIEs in the unit
      llvm::DWARFDie Child = CUDie.getFirstChild();
      while (Child) {
        auto StopOrErr = visitDie(*V, Child);

        if (auto E = StopOrErr.takeError()) {
          llvm::errs() << "Can not visit DIE at " << Child.getOffset() << ": " << E << "\n";
          llvm::consumeError(std::move(E));
        } else if (*StopOrErr) {
          break;
        }
        Child = Child.getSibling();
      }
    }
  }

  return V;
}

DWARFHelper::DWARFHelper(std::string Path)
    : Impl{std::make_unique<DWARFHelper::HelperImpl>(Path)} {}

DWARFHelper::~DWARFHelper() = default;

std::unique_ptr<TypeInfoContainer> DWARFHelper::collectTypeInfo() {
  return Impl->collectTypeInfo();
}

TypeInfoFlags operator|(TypeInfoFlags &L, const TypeInfoFlags &R) {
  return static_cast<TypeInfoFlags>(static_cast<int>(L) | static_cast<int>(R));
}

TypeInfoFlags operator&(TypeInfoFlags &L, const TypeInfoFlags &R) {
  return static_cast<TypeInfoFlags>(static_cast<int>(L) & static_cast<int>(R));
}

TypeInfoFlags& operator|=(TypeInfoFlags &L, const TypeInfoFlags &R) {
  L = L | R;
  return L;
}

std::ostream &operator<<(std::ostream &OS,
                              const Member &M) {
  auto MemberType = M.Type.lock();
  OS << "+" << M.Offset + M.BitOffset / 8 << ":" << M.BitOffset % 8 << "\t"
     << MemberType->TypeName << " " << M.Name;
  unsigned long Size = M.Size ? M.Size : MemberType->Size;
  if (M.BitSize)
    Size += M.BitSize / 8;
  OS << " (" << Size << ":" << M.BitSize << ") ";

  OS << ((MemberType->Flags & kIsAnonymous) ? "A" : "-");
  if (MemberType->Flags & kIsStruct)
    OS << "S";
  else if (MemberType->Flags & kIsUnion)
    OS << "U";
  else
    OS << "-";

  if (MemberType->Flags & kIsFnPtr)
    OS << "F";
  else if (MemberType->Flags & kIsPtr)
    OS << "P";
  else
    OS << "-";
  OS << ((MemberType->Flags & kIsArray) ? "V" : "-");
  return OS;
}

std::ostream &operator<<(std::ostream &OS,
                              const TypeInfo &TI) {
  OS << TI.TypeName << " unqualified='" << TI.BaseName << "'" <<
      " at " << TI.File << ":" << TI.Line <<
      " totalSize=" << TI.Size << "\n";
  if (!TI.AliasNames.empty()) {
    OS << "aka: ";
    for (auto &Name : TI.AliasNames) {
      OS << Name << ",";
    }
    OS << "\n";
  }
  OS << "{\n";
  for (auto &M : TI.Layout) {
    OS << "\t" << M << ",\n";
  }
  OS << "}";
  return OS;
}

} // namespace cheri_benchplot
