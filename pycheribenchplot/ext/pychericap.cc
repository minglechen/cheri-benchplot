#include <cstdint>
#include <iomanip>
#include <limits>
#include <string>
#include <sstream>
#include <pybind11/pybind11.h>

#include "cheri_compressed_cap.h"

namespace py = pybind11;

enum class CC64Perms {
  kPermGlobal = CC64_PERM_GLOBAL,
  kPermExecute = CC64_PERM_EXECUTE,
  kPermLoad = CC64_PERM_LOAD,
  kPermStore = CC64_PERM_STORE,
  kPermLoadCap = CC64_PERM_LOAD_CAP,
  kPermStoreCap = CC64_PERM_STORE_CAP,
  kPermStoreLocal = CC64_PERM_STORE_LOCAL,
  kPermSeal = CC64_PERM_SEAL,
  kPermCInvoke = CC64_PERM_CINVOKE,
  kPermUnseal = CC64_PERM_UNSEAL,
  kPermAccessSysRegs = CC64_PERM_ACCESS_SYS_REGS,
  kPermSetCID = CC64_PERM_SETCID
};

enum class CC128Perms {
  kPermGlobal = CC128_PERM_GLOBAL,
  kPermExecute = CC128_PERM_EXECUTE,
  kPermLoad = CC128_PERM_LOAD,
  kPermStore = CC128_PERM_STORE,
  kPermLoadCap = CC128_PERM_LOAD_CAP,
  kPermStoreCap = CC128_PERM_STORE_CAP,
  kPermStoreLocal = CC128_PERM_STORE_LOCAL,
  kPermSeal = CC128_PERM_SEAL,
  kPermCInvoke = CC128_PERM_CINVOKE,
  kPermUnseal = CC128_PERM_UNSEAL,
  kPermAccessSysRegs = CC128_PERM_ACCESS_SYS_REGS,
  kPermSetCID = CC128_PERM_SETCID
};

enum class CC128mPerms {
  kPermGlobal = CC128M_PERM_GLOBAL,
  kPermExecute = CC128M_PERM_EXECUTE,
  kPermLoad = CC128M_PERM_LOAD,
  kPermStore = CC128M_PERM_STORE,
  kPermLoadCap = CC128M_PERM_LOAD_CAP,
  kPermStoreCap = CC128M_PERM_STORE_CAP,
  kPermStoreLocal = CC128M_PERM_STORE_LOCAL,
  kPermSeal = CC128M_PERM_SEAL,
  kPermCInvoke = CC128M_PERM_CINVOKE,
  kPermUnseal = CC128M_PERM_UNSEAL,
  kPermAccessSysRegs = CC128M_PERM_ACCESS_SYS_REGS,
  kPermSetCID = CC128M_PERM_SETCID,
  kPermExecutive = CC128M_PERM_EXECUTIVE,
  kPermMutableLoad = CC128M_PERM_MUTABLE_LOAD,
};

template<typename CCPerms>
struct CCExtraOps {
};

template<>
struct CCExtraOps<cc64_cap> {
  static constexpr unsigned long MantissaWidth = CC64_MANTISSA_WIDTH;

  static void definePerms(py::handle PyClass) {
    py::enum_<CC64Perms>(PyClass, "Perms", py::arithmetic())
        .value("kPermGlobal", CC64Perms::kPermGlobal)
        .value("kPermExecute", CC64Perms::kPermExecute)
        .value("kPermLoad", CC64Perms::kPermLoad)
        .value("kPermStore", CC64Perms::kPermStore)
        .value("kPermLoadCap", CC64Perms::kPermLoadCap)
        .value("kPermStoreCap", CC64Perms::kPermStoreCap)
        .value("kPermStoreLocal", CC64Perms::kPermStoreLocal)
        .value("kPermSeal", CC64Perms::kPermSeal)
        .value("kPermCInvoke", CC64Perms::kPermCInvoke)
        .value("kPermUnseal", CC64Perms::kPermUnseal)
        .value("kPermAccessSysRegs", CC64Perms::kPermAccessSysRegs)
        .value("kPermSetCID", CC64Perms::kPermSetCID)
        .export_values();
  }

  static void setAddr(cc64_cap &Cap, typename CompressedCap64::addr_t Cursor) {
    cc64_set_addr(&Cap, Cursor);
  }
};

template<>
struct CCExtraOps<cc128_cap> {
  static constexpr unsigned long MantissaWidth = CC128_MANTISSA_WIDTH;

  static void definePerms(py::handle PyClass) {
    py::enum_<CC128Perms>(PyClass, "Perms", py::arithmetic())
        .value("kPermGlobal", CC128Perms::kPermGlobal)
        .value("kPermExecute", CC128Perms::kPermExecute)
        .value("kPermLoad", CC128Perms::kPermLoad)
        .value("kPermStore", CC128Perms::kPermStore)
        .value("kPermLoadCap", CC128Perms::kPermLoadCap)
        .value("kPermStoreCap", CC128Perms::kPermStoreCap)
        .value("kPermStoreLocal", CC128Perms::kPermStoreLocal)
        .value("kPermSeal", CC128Perms::kPermSeal)
        .value("kPermCInvoke", CC128Perms::kPermCInvoke)
        .value("kPermUnseal", CC128Perms::kPermUnseal)
        .value("kPermAccessSysRegs", CC128Perms::kPermAccessSysRegs)
        .value("kPermSetCID", CC128Perms::kPermSetCID)
        .export_values();
  }

  static void setAddr(cc128_cap &Cap, typename CompressedCap128::addr_t Cursor) {
    cc128_set_addr(&Cap, Cursor);
  }
};

template<>
struct CCExtraOps<cc128m_cap> {
  static constexpr unsigned long MantissaWidth = CC128M_MANTISSA_WIDTH;

  static void definePerms(py::handle PyClass) {
    py::enum_<CC128mPerms>(PyClass, "Perms", py::arithmetic())
        .value("kPermGlobal", CC128mPerms::kPermGlobal)
        .value("kPermExecute", CC128mPerms::kPermExecute)
        .value("kPermLoad", CC128mPerms::kPermLoad)
        .value("kPermStore", CC128mPerms::kPermStore)
        .value("kPermLoadCap", CC128mPerms::kPermLoadCap)
        .value("kPermStoreCap", CC128mPerms::kPermStoreCap)
        .value("kPermStoreLocal", CC128mPerms::kPermStoreLocal)
        .value("kPermSeal", CC128mPerms::kPermSeal)
        .value("kPermCInvoke", CC128mPerms::kPermCInvoke)
        .value("kPermUnseal", CC128mPerms::kPermUnseal)
        .value("kPermAccessSysRegs", CC128mPerms::kPermAccessSysRegs)
        .value("kPermSetCID", CC128mPerms::kPermSetCID)
        .value("kPermExecutive", CC128mPerms::kPermExecutive)
        .value("kPermMutableLoad", CC128mPerms::kPermMutableLoad)
        .export_values();
  }

  static void setAddr(cc128m_cap &Cap, typename CompressedCap128m::addr_t Cursor) {
    cc128m_set_addr(&Cap, Cursor);
  }
};

template<typename CC, typename CCOps>
void defineCap(py::handle M, const char *Name) {

  using CCExtra = CCExtraOps<CC>;
  using AddrT = typename CCOps::addr_t;
  using LengthT = typename CCOps::length_t;

  auto PyCap =
      py::class_<CC>(M, Name)
      .def(py::init<>())
      .def_readwrite("tag", &CC::cr_tag)
      .def_readonly("bounds_valid", &CC::cr_tag)
      .def_readwrite("exponent", &CC::cr_exp)
      .def("base", &CC::base)
      .def("address", &CC::address)
      .def("offset", &CC::offset)
      .def("top", &CC::top)
      .def("top64", &CC::top64)
      .def("length", &CC::length)
      .def("length64", &CC::length64)
      .def("software_permissions", &CC::software_permissions)
      .def("permissions", &CC::permissions)
      .def("type", &CC::type)
      .def("is_sealed", &CC::is_sealed)
      .def("reserved_bits", &CC::reserved_bits)
      .def("flags", &CC::flags)
      .def("__eq__", [](const CC &Left, const CC &Right) {
        return Left == Right;
      })
      .def("__str__", [](const CC &Cap) {
        std::stringstream SS;

        SS << std::hex << Cap.address() << " [" << Cap.base() << ", " <<
            Cap.top64() << "]";

        return SS.str();
      })
      .def("__repr__", [](const CC &Cap) {
        std::stringstream SS;

        SS << std::boolalpha <<
            "Valid: " << static_cast<bool>(Cap.cr_tag) << "\n" <<
            "Sealed: " << static_cast<bool>(Cap.is_sealed()) << "\n" << std::hex <<
            "Perms: 0x" << Cap.permissions() << "\n" <<
            "User Perms: 0x" << Cap.software_permissions() << "\n" <<
            "Base: 0x" << Cap.base() << "\n" <<
            "Addr: 0x" << Cap.address() << "\n" <<
            "Length: 0x" << Cap.length64() <<
            (Cap.length() > UINT64_MAX ? " (overflow)\n" : "\n") <<
            "Top: 0x" << Cap.top64() <<
            (Cap.top() > UINT64_MAX ? " (overflow)\n" : "\n") <<
            "OType: 0x" << Cap.type() << "\n" <<
            "Flags: 0x" << Cap.flags() << "\n";
        return SS.str();
      })
      .def("compress_raw", [](const CC &Cap) {
        return CCOps::compress_raw(Cap);
      })
      .def("compress_mem", [](const CC &Cap) {
        return CCOps::compress_mem(Cap);
      })
      .def("setbounds", [](CC &Cap, LengthT Length) {
        return CCOps::setbounds(&Cap, Length);
      })
      .def("setaddr", [](CC &Cap, AddrT Cursor) {
        CCExtra::setAddr(Cap, Cursor);
      })
      .def_static("make_max_perms_cap", [](AddrT Base, AddrT Cursor, LengthT Top) {
        return CCOps::make_max_perms_cap(Base, Cursor, Top);
      })
      .def_static("make_max_bounds_cap", [](AddrT Cursor) {
        LengthT MaxTop = std::numeric_limits<AddrT>::max() + 1;
        CC Cap = CCOps::make_max_perms_cap(0, 0, MaxTop);
        CCExtra::setAddr(Cap, Cursor);
        return Cap;
      })
      .def_static("make_null_derived_cap", [](AddrT Addr) {
        return CCOps::make_null_derived_cap(Addr);
      })
      .def_static("decompress_raw", [](AddrT Pesbt, AddrT Cursor, bool Tag) {
        return CCOps::decompress_raw(Pesbt, Cursor, Tag);
      })
      .def_static("decompress_mem", [](AddrT Pesbt, AddrT Cursor, bool Tag) {
        return CCOps::decompress_mem(Pesbt, Cursor, Tag);
      })
      .def_static("representable_mask", [](AddrT Len) {
        return CCOps::representable_length(Len);
      })
      .def_static("representable_mask", [](AddrT Len) {
        return CCOps::representable_mask(Len);
      })
      .def_static("get_mantissa_width", []() {
        return CCExtra::MantissaWidth;
      });
  CCExtra::definePerms(PyCap);
}

PYBIND11_MODULE(pychericap, M)
{
  defineCap<cc64_cap, CompressedCap64>(M, "CompressedCap64");
  defineCap<cc128_cap, CompressedCap128>(M, "CompressedCap128");
  defineCap<cc128m_cap, CompressedCap128m>(M, "CompressedCap128m");
}
