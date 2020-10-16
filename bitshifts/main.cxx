#include <bitset>
#include <iostream>

constexpr unsigned int digits = 32;

template <typename IntType>
void shift_bits(IntType i, int ls, int rs, std::string int_type_name) {
  IntType il = i << ls, ir = i >> rs;

  std::cout << "***************" << std::endl;

  std::cout << "Left bit-shift of i = " << i << std::endl
            << "   by   " << ls << "   is   " << std::endl
            << "(" << int_type_name << ")  " << il << std::endl
            << "(int)     " << static_cast<int>(il) << std::endl;

  std::cout << "bits change to" << std::endl
            << std::bitset<digits>(i) << std::endl
            << std::bitset<digits>(il) << std::endl;

  /*
   *
   */

  //  std::cout << std::endl
  //            << "Right bit-shift of i = " << i << std::endl
  //            << "   by   " << rs << "   is   " << std::endl
  //            << "(" << int_type_name << ")  " << ir << std::endl
  //            << "(int)     " << static_cast<int>(ir) << std::endl;
  //
  //  std::cout << "bits (" << digits << ") change to" << std::endl
  //            << std::bitset<digits>(i) << std::endl
  //            << std::bitset<digits>(ir) << std::endl;
  //
  //  std::cout << "i = " << static_cast<int>(i) << std::endl;

  std::cout << "***************" << std::endl << std::endl;
}

int main() {
  int8_t i8 = 1;
  int32_t i = 1;
  int ls = 31, rs = 1;

  //  shift_bits(i8, ls, rs, "int8_t");
  shift_bits(i, ls, rs, "int32_t");

  return 0;
}
