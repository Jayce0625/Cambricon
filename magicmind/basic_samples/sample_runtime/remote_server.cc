/*************************************************************************
 * Copyright (C) [2020-2023] by Cambricon, Inc.
 * Description:
 *************************************************************************/
#include "mm_remote.h"
#include "common/macros.h"
#include "common/logger.h"

#include <csignal>

static magicmind::IRpcServer* server = nullptr;

static void SIGINTHandler( int signum )
{
  SLOG(INFO) << "Interrupt signal (" << signum << ") received.";
  CHECK_STATUS(server->Shutdown());
  CHECK_STATUS(server->Destroy());
  exit(signum);
}

int main() {
  signal(SIGINT, SIGINTHandler); 
  // Create server.
  std::string port = "9009";  // default port
  server = magicmind::CreateRpcServer(port);
  CHECK_VALID(server);
  CHECK_STATUS(server->HandleRPCsLoop());
  return 0;
}