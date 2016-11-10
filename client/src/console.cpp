#include "console.h"
#include "client_socket.h"
#include <fstream>
#include <stdio.h>
#include <iostream>
#include <readline/readline.h>
#include <readline/history.h>

Console::Console(): mRunning(false),
    mUnknownCommandMessage("Unknown command") {

    addQuit();
    addHelp();
    addQuery();
    addMan();

    // addTask([](){cout << "-=-=-=-=-=-=-=-" << endl;});

    int commands_size = mCommands.size();
    tab_commands = (char**) new char*[reserved_tab_command_size];
    for (int i = 0; i < commands_size; i++) {
        const string command_name = mCommands[i].getName();
        tab_commands[i] = (char *) new char[command_name.size() + 1];
        memcpy(tab_commands[i], command_name.c_str(), command_name.size());
        tab_commands[i][command_name.size()] = '\0';
    }

    rl_readline_name = "wukong_client";
    rl_attempted_completion_function = command_completion;
}

Console::~Console() {
    delete client_socket;
    int commands_size = mCommands.size();
    for (int i = 0; i < commands_size; i++) {
        delete [] tab_commands[i];
    }
    delete [] tab_commands;
    tab_commands = NULL;
}

Console *Console::getInstance() {
    if (!instance) {
        cout << "Console construction!" << endl;
        cout << "Welcome to use wukong-client" << endl;
        cout << "----------------------------------------------------------------------" << endl;
        instance = new Console();
    }
    return instance;
}

void Console::setServer(char *filename) {
    ifstream server_file(filename);
    if (!server_file) {
        cout << "File " << filename << " not exist" << endl;
        return;
    }
    string broker_name;
    int broker_port;
    server_file >> broker_name >> broker_port;
    client_socket = new Client_Socket(broker_name, broker_port);
    cout << "set server: " << broker_name << ":" << broker_port << endl;
}
void Console::deleteInstance() {
    if (instance) {
        cout << "Console destruction!" << endl;
        delete instance;
        instance = NULL;
    }
}

void Console::run() {
    if (!client_socket) {
        cout << "error:no server determined" << endl;
        exit(-1);
    }
    mRunning = true;
    while (mRunning) {
        char* buf;

        buf = readline("-> ");
        if (!buf)continue;
        if (!(*buf)) {
            free(buf);
            continue;
        }
        add_history(buf);
        string cmd(buf);
        free(buf);
        string commandName = getCommandName(cmd);
        string commandOptions = getCommandOptions(cmd);

        bool found = false;
        for (size_t i = 0; i < mCommands.size(); i++) {
            if (mCommands[i].isCommand(commandName)) {
                found = true;
                mCommands[i].execute(commandOptions);
            }
        }
        if (!found) {
            cout << mUnknownCommandMessage << endl;
        }

        for (auto task : mTasks) {
            if (task) {
                task();
            }
        }

        if (mRunning) {
            cout << endl;
        }
    }
}

void Console::stop() {
    mRunning = false;
}


string Console::getCommandName(string const &input) {
    string commandName;
    size_t found = input.find_first_of(" ");
    if (found != string::npos) {
        commandName = input.substr(0, found);
    } else {
        commandName = input;
    }
    return commandName;
}

string Console::getCommandOptions(string const &input) {
    string commandOptions = "";
    size_t found = input.find_first_of(" ");
    if (found != string::npos) {
        commandOptions = input.substr(found + 1);
    }
    return commandOptions;
}
char *Console::command_generator(const char *text, int state) {
    const char *name;
    static int list_index, len;
    if (!state) {
        list_index = 0;
        len = strlen (text);
    }

    while (name = tab_commands[list_index]) {
        list_index++;
        if (strncmp (name, text, len) == 0)
            return strdup(name);
    }

    return ((char *)NULL);
}

char** Console::command_completion (const char *text, int start, int end) {
    char **matches = NULL;

    if (start == 0)
        matches = rl_completion_matches (text, command_generator);

    return (matches);
}

void Console::addCommand(Command const &command) {
    mCommands.push_back(command);
}

void Console::removeCommand(string const &command) {
    for (size_t i = 0; i < mCommands.size();) {
        if (mCommands[i].isCommand(command)) {
            mCommands.erase(i + mCommands.begin());
        } else {
            i++;
        }
    }
}

void Console::addTask(function<void()> task) {
    mTasks.push_back(task);
}

void Console::addQuit() {
    Command quit;
    quit.setName("quit");
    quit.setHelp("Quit the console");
    quit.setManual("quit\nQuit the console properly\nNo options available");
    quit.setFunction([this](Command::OptionSplit options) {stop();});
    addCommand(quit);
}

void Console::addHelp() {
    Command help;
    help.setName("help");
    help.setHelp("See the list of commands with short description");
    help.setManual("help\nSee the list of commands with short description\nNo options available");
    help.setFunction([this](Command::OptionSplit options) {
        for (auto command : mCommands) {
            cout << command.getName() << " : " << command.getHelp() << endl;
        }
    });
    addCommand(help);
}

void Console::addMan() {
    Command man;
    man.setName("man");
    man.setHelp("See detailed information for a command");
    man.setManual("man\nSee detailed information for a command\nOptions : -c commandName");
    man.setFunction([this](Command::OptionSplit options) {
        if (options["c"].size() == 1) {
            string commandName = options["c"][0];
            for (auto command : mCommands) {
                if (command.isCommand(commandName)) {
                    cout << command.getManual() << endl;
                }
            }
        } else {
            cout << "Error with the options, ex : man -c man" << endl;
        }
    });
    addCommand(man);
}


void Console::addQuery() {
    Command query;
    query.setName("query"); // give it a name
    query.addAlias("q"); // add an alias
    query.setHelp("Execute the query"); // // short desc
    query.setManual("query\nExecute the query\nOptions :\n-l : console input (default)\n-f : file input"); // long desc for man
    query.setFunction([this](Command::OptionSplit options) {
        CS_Request request;
        request.type = QUERY_TYPE;
        auto itr = options.find("l");
        if (itr != options.end()) {
            string str;
            while (getline(cin, str)) {
                request.content += str + '\n';
                if (str == "}")break;
            }
        } else if ((itr = options.find("f")) != options.end()) {
            if (itr->second.size() >= 1) {
                request.use_file = true;
                request.content = itr->second[0];
            }
        } else {
            cout << "error:" << endl << endl << "wrong format, please see Man for more details";
            return;
        }
        cout << request.content << endl;
        client_socket->send_request(request);
        CS_Reply reply = client_socket->recv_reply();
        if (reply.type == ERROR) {
            cout << "error:" << endl << endl << reply. content << endl;
        } else {
            cout << "reply:" << endl << endl << reply.column << endl;
            reply.print();
        }
    });
    addCommand(query); // add the command to the console
}
