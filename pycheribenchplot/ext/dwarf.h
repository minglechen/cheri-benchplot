

#pragma once

#include <cstddef>
#include <iterator>
#include <unordered_map>
#include <memory>
#include <optional>
#include <ostream>
#include <set>
#include <string>
#include <vector>

namespace cheri_benchplot {

enum TypeInfoFlags {
  kNone = 0,
  kIsAnonymous = 1,
  kIsStruct = 1 << 1,
  kIsUnion = 1 << 2,
  kIsPtr = 1 << 3,
  kIsFnPtr = 1 << 4,
  kIsArray = 1 << 5,
  kIsConst = 1 << 6,
  kIsDecl = 1 << 7
};

TypeInfoFlags operator|(TypeInfoFlags &L, const TypeInfoFlags &R);
TypeInfoFlags operator&(TypeInfoFlags &L, const TypeInfoFlags &R);
TypeInfoFlags& operator|=(TypeInfoFlags &L, const TypeInfoFlags &R);

struct TypeInfo;

struct Member {
  std::string Name;
  unsigned long Line;
  unsigned long Offset;
  unsigned long Size;
  unsigned long BitOffset;
  unsigned long BitSize;
  std::weak_ptr<TypeInfo> Type;
};

struct TypeInfo {
  // Unique identifier for the typeinfo, this is the offset in the debug section.
  unsigned long Handle;
  // File where the type is defined
  std::string File;
  // Line where the type is defined
  unsigned long Line;
  // Reported aggregate size
  unsigned long Size;
  // Flags identifying the type
  TypeInfoFlags Flags;
  // For arrays, this is the number of items. 0 if it is a flexible array.
  unsigned long ArrayItems;
  // Unmodified base type name
  std::string BaseName;
  // Type name modified by CV-qualifiers pointers and references.
  std::string TypeName;
  // For composite types, member descriptions
  std::vector<Member> Layout;
  // Alias names from typedefs
  std::set<std::string> AliasNames;

  TypeInfo() = default;
};

struct TypeInfoContainer {
  using CompositeTypeMap = std::unordered_map<unsigned long, std::shared_ptr<TypeInfo>>;

  struct Iterator {
    using iterator_category = std::forward_iterator_tag;
    using difference_type = std::ptrdiff_t;
    using value_type = TypeInfo;
    using pointer = TypeInfo*;
    using reference = TypeInfo&;

    explicit Iterator(CompositeTypeMap::const_iterator &&Iter)
        : Inner(Iter) {}

    Iterator& operator++() {
      ++Inner;
      return *this;
    }
    Iterator operator++(int) {
      Iterator Retval = *this;
      ++(*this);
      return Retval;
    }
    bool operator==(Iterator Other) const {
      return Inner == Other.Inner;
    }
    bool operator!=(Iterator Other) const {
      return Inner != Other.Inner;
    }
    reference operator*() const {
      return *(Inner->second);
    }
   private:
    CompositeTypeMap::const_iterator Inner;
  };

  TypeInfoContainer() = default;
  virtual ~TypeInfoContainer() = default;

  virtual std::optional<TypeInfo> findComposite(unsigned long Handle) const = 0;
  virtual Iterator beginComposite() const = 0;
  virtual Iterator endComposite() const = 0;
};

/// Top level DWARF parser
class DWARFHelper {
  class HelperImpl;

public:
  DWARFHelper(std::string Path);
  ~DWARFHelper();

  // int getABIPointerSize() const;
  // int getABICapabilitySize() const;
  std::unique_ptr<TypeInfoContainer> collectTypeInfo();

private:
  std::unique_ptr<HelperImpl> Impl;
};

std::ostream &operator<<(std::ostream &OS, const Member &M);
std::ostream &operator<<(std::ostream &OS, const TypeInfo &TI);

} // namespace cheri_benchplot

extern "C" {};
