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

#ifndef COMMAND_HPP
#define COMMAND_HPP

#include <functional>
#include <iostream>
#include <map>
#include <string>
#include <vector>

class Command {
public:
    Command();

    typedef std::map<std::string, std::vector<std::string>> OptionSplit;
    typedef std::function<void(OptionSplit options)> CommandFunction;

    std::string getName() const;
    std::string getHelp() const;
    std::string getManual() const;

    void execute(std::string options = "");

    void setName(std::string const &name);
    void setHelp(std::string const &help);
    void setManual(std::string const &manual);
    void setFunction(CommandFunction function);

    bool hasAlias(std::string const &alias);
    void addAlias(std::string const &alias);

    bool isCommand(std::string const &name);

    static OptionSplit splitOptions(std::string options);

private:
    std::string mName;
    std::string mHelp;
    std::string mManual;
    std::vector<std::string> mAlias;
    CommandFunction mFunction;
};

#endif // COMMAND_HPP