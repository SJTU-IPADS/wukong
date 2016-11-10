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