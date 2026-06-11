# Usage: cmake -DINPUT=<file> -DOUTPUT=<header> -DVAR=<identifier> -P EmbedFile.cmake
# Embeds INPUT as `inline constexpr unsigned char VAR[]` in namespace mn::web.
if(NOT INPUT OR NOT OUTPUT OR NOT VAR)
    message(FATAL_ERROR "EmbedFile.cmake needs -DINPUT, -DOUTPUT, -DVAR")
endif()
file(READ "${INPUT}" hex HEX)
string(REGEX REPLACE "([0-9a-f][0-9a-f])" "0x\\1," bytes "${hex}")
file(WRITE "${OUTPUT}" "// Generated from ${INPUT} - do not edit.
#pragma once
#include <cstddef>
namespace mn::web {
inline constexpr unsigned char ${VAR}[] = {${bytes}};
inline constexpr size_t ${VAR}_len = sizeof(${VAR});
} // namespace mn::web
")
