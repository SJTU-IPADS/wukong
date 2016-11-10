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