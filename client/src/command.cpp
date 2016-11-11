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

#include "command.h"

Command::Command() {
    mName = "";
    mHelp = "";
    mManual = "";
}

std::string Command::getName() const {
    return mName;
}

std::string Command::getHelp() const {
    return mHelp;
}

std::string Command::getManual() const {
    return mManual;
}

void Command::execute(std::string options) {
    if (mFunction)
        mFunction(splitOptions(options));
    else
        std::cout << "Command not properly set" << std::endl;
}

void Command::setName(std::string const &name) {
    mName = name;
}

void Command::setHelp(std::string const &help) {
    mHelp = help;
}

void Command::setManual(std::string const &manual) {
    mManual = manual;
}

void Command::setFunction(Command::CommandFunction function) {
    mFunction = function;
}

bool Command::hasAlias(std::string const &alias) {
    for (auto a : mAlias)
        if (a == alias)
            return true;
    return false;
}

void Command::addAlias(std::string const& alias) {
    mAlias.push_back(alias);
}

bool Command::isCommand(std::string const& name) {
    return (name == mName || hasAlias(name));
}

Command::OptionSplit Command::splitOptions(std::string options) {
    OptionSplit split;
    if (options != "") {
        bool open = false;
        bool readParam = true;
        bool readOption = false;
        bool readArgs = false;
        std::size_t readPos = 0;
        std::string actualOption = "";
        for (std::size_t i = 0; i < options.size(); i++) {
            if (options[i] == ' ' && readParam) {
                if (readPos == 0) {
                    split["param"].push_back(options.substr(readPos, i - readPos));
                } else {
                    split["param"].push_back(options.substr(readPos + 1, i - readPos - 1));
                }
                readPos = i;
            }
            else if (options[i] == '-' && !open) {
                readArgs = false;
                readParam = false;
                readOption = true;
                readPos = i;
            } else if (options[i] == ' ' && readOption == true) {
                actualOption = options.substr(readPos + 1, i - readPos - 1);
                split[actualOption] = {};
                readOption = false;
                readArgs = true;
                readPos = i;
            } else if (options[i] == ' ' && readArgs && !open) {
                std::string arg = options.substr(readPos + 1, i - readPos - 1);
                if (arg != "") {
                    if (arg[0] == '\"' && arg[arg.size() - 1] == '\"') {
                        arg = arg.substr(1, arg.size() - 2);
                    }
                    split[actualOption].push_back(arg);
                }
                readPos = i;
            } else if (options[i] == '\"' && readArgs) {
                open = !open;
            }
        }
        if (readParam) {
            split["param"].push_back(options.substr(readPos + ((readPos == 0) ? 0 : 1)));
        } else if (readOption) {
            split[options.substr(readPos + 1)] = {};
        } else if (readArgs) {
            std::string arg = options.substr(readPos + 1);
            if (arg != "") {
                if (arg[0] == '\"' && arg[arg.size() - 1] == '\"') {
                    arg = arg.substr(1, arg.size() - 2);
                }
                split[actualOption].push_back(arg);
            }
        }
    }
    return split;
}