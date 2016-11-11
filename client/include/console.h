/*
 * Copyright (c) 2016 Shanghai Jiao Tong University.
 *     All rights reserved.
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing,
 *  software distributed under the License is distributed on an "AS
 *  IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
 *  express or implied.  See the License for the specific language
 *  governing permissions and limitations under the License.
 *
 * For more about this software visit:
 *
 *      http://ipads.se.sjtu.edu.cn/projects/wukong.html
 *
 */

#ifndef CONSOLE_HPP
#define CONSOLE_HPP

#include <iostream>
#include <string>
#include <vector>
#include "command.h"
#include "client_socket.h"

using namespace std;

class Console {
public:
    Console();
    ~Console();
    static Console* getInstance();
    static void deleteInstance();
    void run();
    void stop();
    void setServer(char* filename);

private:
    bool mRunning;
    string mUnknownCommandMessage;
    vector<Command> mCommands;
    vector<function<void()>> mTasks;
    Client_Socket* client_socket;
    static Console* instance;
    static char** tab_commands;

    static string getCommandName(string const& input);
    static string getCommandOptions(string const& input);
    static char* command_generator(const char *text, int state);
    static char** command_completion (const char *text, int start, int end);

    const int reserved_tab_command_size = 100;

    void addCommand(Command const& command);
    void removeCommand(string const& command);
    void addTask(function<void()> task);

    void addQuit();
    void addHelp();
    void addMan();
    void addQuery();
};

#endif // CONSOLE_HPP