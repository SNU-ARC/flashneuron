#pragma once

#ifndef _WIN32
#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

#include <sys/types.h>
#include <cstring>

#include <condition_variable>
#include <mutex>
#include <string>
#include <system_error>
#include <vector>

namespace c10d {
namespace test {

class Semaphore {
 public:
  void post(int n = 1) {
    std::unique_lock<std::mutex> lock(m_);
    n_ += n;
    cv_.notify_all();
  }

  void wait(int n = 1) {
    std::unique_lock<std::mutex> lock(m_);
    while (n_ < n) {
      cv_.wait(lock);
    }
    n_ -= n;
  }

 protected:
  int n_ = 0;
  std::mutex m_;
  std::condition_variable cv_;
};

#ifdef _WIN32
std::string autoGenerateTmpFilePath() {
  char tmp[L_tmpnam_s];
  errno_t err;
  err = tmpnam_s(tmp, L_tmpnam_s);
  if (err != 0)
  {
    throw std::system_error(errno, std::system_category());
  }
  return std::string(tmp);
}

std::string tmppath() {
  const char* tmpfile = getenv("TMPFILE");
  if (tmpfile) {
    return std::string(tmpfile);
  }
  else {
    return autoGenerateTmpFilePath();
  }
}
#else
std::string tmppath() {
  // TMPFILE is for manual test execution during which the user will specify
  // the full temp file path using the environmental variable TMPFILE
  const char* tmpfile = getenv("TMPFILE");
  if (tmpfile) {
    return std::string(tmpfile);
  }

  const char* tmpdir = getenv("TMPDIR");
  if (tmpdir == nullptr) {
    tmpdir = "/tmp";
  }

  // Create template
  std::vector<char> tmp(256);
  auto len = snprintf(tmp.data(), tmp.size(), "%s/testXXXXXX", tmpdir);
  tmp.resize(len);

  // Create temporary file
  auto fd = mkstemp(&tmp[0]);
  if (fd == -1) {
    throw std::system_error(errno, std::system_category());
  }
  close(fd);
  return std::string(tmp.data(), tmp.size());
}
#endif

bool isTSANEnabled() {
  auto s = std::getenv("PYTORCH_TEST_WITH_TSAN");
  return s && strcmp(s, "1") == 0;
}
struct TemporaryFile {
  std::string path;

  TemporaryFile() {
    path = tmppath();
  }

  ~TemporaryFile() {
    unlink(path.c_str());
  }
};

#ifndef _WIN32
struct Fork {
  pid_t pid;

  Fork() {
    pid = fork();
    if (pid < 0) {
      throw std::system_error(errno, std::system_category(), "fork");
    }
  }

  ~Fork() {
    if (pid > 0) {
      kill(pid, SIGKILL);
      waitpid(pid, nullptr, 0);
    }
  }

  bool isChild() {
    return pid == 0;
  }
};
#endif

} // namespace test
} // namespace c10d
