#pragma once

struct ConfigFlags {
  bool recording = true;
  bool detection = true;
  bool effects = true;
  bool sound = true;
  bool cleanup = true;
  bool animation = true;
  bool face = true;
  bool debug_server = false;
  int debug_port = 8080;
  bool help = false;
};

ConfigFlags parseCmdlineArgs(int argc, char *argv[]);
void printHelp(const char *progname);