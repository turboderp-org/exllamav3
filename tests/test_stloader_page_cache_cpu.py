"""Exercise the production positioned-read function without Torch or a GPU."""
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import unittest


@unittest.skipUnless(sys.platform == "linux" and shutil.which("g++"), "Linux and g++ required")
class PageCacheTests(unittest.TestCase):
    def test_buffered_reads_release_cache_unless_kept(self):
        root = Path(__file__).resolve().parents[1]
        source = (root / "exllamav3/exllamav3_ext/stloader.cpp").read_text()
        read_range = re.search(r"static bool read_range\([^\n]+\n\{.*?^\}", source, re.M | re.S)
        self.assertIsNotNone(read_range)
        program = r'''
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <vector>
''' + read_range[0] + r'''
static void require(bool ok, const char* message) {
    if (!ok) { std::fprintf(stderr, "%s\n", message); std::exit(1); }
}
int main(int argc, char** argv) {
    const size_t size = 2 * 1024 * 1024;
    const size_t page = sysconf(_SC_PAGESIZE);
    FILE* file = std::fopen(argv[1], "w+b");
    require(file != nullptr, "open failed");
    int fd = fileno(file);
    std::vector<uint8_t> expected(size), output(size + page);
    for (size_t i = 0; i < size; ++i) expected[i] = i % 251;
    require(std::fwrite(expected.data(), 1, size, file) == size, "write failed");
    require(std::fflush(file) == 0 && fsync(fd) == 0, "sync failed");
    void* mapping = mmap(nullptr, size, PROT_READ, MAP_SHARED, fd, 0);
    require(mapping != MAP_FAILED, "mmap failed");
    require(pread(fd, output.data(), size, 0) == (ssize_t)size, "warm read failed");
    std::vector<unsigned char> pages(size / page);
    require(mincore(mapping, size, pages.data()) == 0, "mincore failed");
    size_t warm = 0;
    for (auto state : pages) warm += bool(state & 1);
    require(warm == pages.size(), "fixture did not populate the page cache");

    int errnum; bool eof;
    require(read_range(file, 128, size - 256, output.data(), errnum, eof), "read failed");
    require(std::memcmp(output.data(), expected.data() + 128, size - 256) == 0, "bytes changed");
    require(mincore(mapping, size, pages.data()) == 0, "mincore after read failed");
    size_t retained = 0;
    for (auto state : pages) retained += bool(state & 1);
    std::printf("warm=%zu retained=%zu\n", warm, retained);
    if (std::strcmp(argv[2], "keep") == 0)
        require(retained == warm, "explicit cache retention was ignored");
    else
        require(retained <= 2, "consumed full pages remain cached");

    require(!read_range(file, 0, size + page, output.data(), errnum, eof), "EOF read succeeded");
    require(eof && errnum == 0, "EOF semantics changed");
    require(std::memcmp(output.data(), expected.data(), size) == 0, "short-read bytes changed");
    munmap(mapping, size);
    std::fclose(file);
}
'''
        # Use the checkout filesystem: tmpfs cannot evict its file data.
        with tempfile.TemporaryDirectory(dir=root) as directory:
            directory = Path(directory)
            cpp, binary = directory / "read.cpp", directory / "read"
            cpp.write_text(program)
            subprocess.run(["g++", "-std=c++17", "-O2", str(cpp), "-o", str(binary)], check=True)
            for keep in (True, False):
                with self.subTest(keep=keep):
                    env = os.environ.copy()
                    env.pop("EXL3_DROP_PAGE_CACHE", None)
                    if not keep:
                        env["EXL3_DROP_PAGE_CACHE"] = "1"
                    result = subprocess.run([str(binary), str(directory / "shard"), "keep" if keep else "drop"],
                                            env=env, capture_output=True, text=True)
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
