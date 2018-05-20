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
 *      http://ipads.se.sjtu.edu.cn/projects/wukong
 *
 */

//---------------------------------------------------------------------------
// RDF-3X
// (c) 2008 Thomas Neumann. Web site: http://www.mpi-inf.mpg.de/~neumann/rdf3x
//
// This work is licensed under the Creative Commons
// Attribution-Noncommercial-Share Alike 3.0 Unported License. To view a copy
// of this license, visit http://creativecommons.org/licenses/by-nc-sa/3.0/
// or send a letter to Creative Commons, 171 Second Street, Suite 300,
// San Francisco, California, 94105, USA.
//---------------------------------------------------------------------------

#pragma once

#include <map>
#include <string>
#include <vector>
#include <memory>
#include <cstdlib>
#include <iostream>

#include "SPARQLLexer.hpp"
#include "type.hpp"

using namespace std;

class SPARQLLexer;

/// A parser for SPARQL input
class SPARQLParser {
public:
    /// A parsing exception
    struct ParserException {
        /// The message
        std::string message;

        /// Constructor
        ParserException(const std::string &message) : message(message) { }
        /// Constructor
        ParserException(const char *message) : message(message) { }
        /// Destructor
        ~ParserException() { }
    };

    /// An element in a graph pattern
    struct Element {
        /// Possible types
        enum Type { Variable, Literal, IRI, Template, Predicate };
        /// Possible sub-types for literals
        enum SubType { None, CustomLanguage, CustomType };
        /// The type
        Type type;
        /// The sub-type
        SubType subType;
        /// The value of the sub-type
        std::string subTypeValue;
        /// The literal value
        std::string value;
        /// The id for variables
        unsigned id;
    };

    /// A graph pattern
    struct Pattern {
        /// The entires
        Element subject, predicate, object;
        /// Direction
        dir_t direction = OUT;
        /// Constructor
        Pattern(Element subject, Element predicate, Element object)
            : subject(subject), predicate(predicate), object(object) { }
        /// Destructor
        ~Pattern() { }
    };

    /// A filter entry
    struct Filter {
        /// Possible types
        enum Type {
            Or, And, Equal, NotEqual, Less, LessOrEqual, Greater,
            GreaterOrEqual, Plus, Minus, Mul, Div, Not, UnaryPlus, UnaryMinus,
            Literal, Variable, IRI, Function, ArgumentList, Builtin_str,
            Builtin_lang, Builtin_langmatches, Builtin_datatype, Builtin_bound,
            Builtin_sameterm, Builtin_isiri, Builtin_isblank, Builtin_isliteral,
            Builtin_regex, Builtin_in
        };

        /// The type
        Type type;
        /// Input arguments
        Filter *arg1, *arg2, *arg3;
        /// The value (for constants)
        std::string value;
        /// The type (for constants)
        std::string valueType;
        /// Possible subtypes or variable ids
        int valueArg;

        /// Constructor
        Filter() : arg1(0), arg2(0), arg3(0), valueArg(0) { }
        /// Copy-Constructor
        Filter(const Filter &other)
            : type(other.type), arg1(0), arg2(0), arg3(0),
              value(other.value), valueType(other.valueType),
              valueArg(other.valueArg) {
            if (other.arg1)
                arg1 = new Filter(*other.arg1);
            if (other.arg2)
                arg2 = new Filter(*other.arg2);
            if (other.arg3)
                arg3 = new Filter(*other.arg3);
        }
        /// Destructor
        ~Filter() {
            delete arg1;
            delete arg2;
            delete arg3;
        }

        /// Assignment
        Filter &operator=(const Filter &other) {
            if (this != &other) {
                type = other.type;
                delete arg1;
                if (other.arg1)
                    arg1 = new Filter(*other.arg1);
                else
                    arg1 = 0;

                delete arg2;
                if (other.arg2)
                    arg2 = new Filter(*other.arg2);
                else
                    arg2 = 0;

                delete arg3;
                if (other.arg3)
                    arg3 = new Filter(*other.arg3);
                else
                    arg3 = 0;

                value = other.value;
                valueType = other.valueType;
                valueArg = other.valueArg;
            }
            return *this;
        }
    };

    /// A group of patterns
    struct PatternGroup {
        /// The patterns
        std::vector<Pattern> patterns;
        /// The filter conditions
        std::vector<Filter> filters;
        /// The optional parts
        std::vector<PatternGroup> optional;
        /// The union parts
        std::vector<PatternGroup> unions;
    };

    /// The projection modifier
    enum ProjectionModifier { Modifier_None, Modifier_Distinct,
                              Modifier_Reduced, Modifier_Count,
                              Modifier_Duplicates
                            };
    /// Sort order
    struct Order {
        /// Variable id
        int id;
        /// Desending
        bool descending;
    };

private:
    /// The lexer
    SPARQLLexer &lexer;
    /// The registered prefixes
    std::map<std::string, std::string> prefixes;
    /// The named variables
    std::map<std::string, int> namedVariables;
    /// The total variable count
    unsigned variableCount;
    /// The named variable count
    unsigned namedVariableCount;

    /// The projection modifier
    ProjectionModifier projectionModifier;
    /// The projection clause
    std::vector<int> projection;
    /// The pattern
    PatternGroup patterns;
    /// The sort order
    std::vector<Order> order;
    /// The result limit, -1 means no limit
    int limit;
    /// The result offset
    unsigned offset;
    // indicate if custom grammar is in use
    bool usingCustomGrammar;
    bool corun_enabled;
    int corun_step;
    int fetch_step;

    /// Lookup or create a named variable
    int nameVariable(const std::string &name) {
        if (namedVariables.count(name))
            return namedVariables[name];

        variableCount++;
        int result = ++namedVariableCount;
        namedVariables[name] = -result;
        return -result;
    }

    /// Parse an RDF literal
    void parseRDFLiteral(string &value, Element::SubType &subType, string &valueType) {
        if (lexer.getNext() != SPARQLLexer::String)
            throw ParserException("literal expected");
        subType = Element::None;
        value = lexer.getLiteralValue();
        if (value.find('\\') != string::npos) {
            string v; v.swap(value);
            for (string::const_iterator iter = v.begin(), limit = v.end(); iter != limit; ++iter) {
                char c = (*iter);
                if (c == '\\') {
                    if ((++iter) == limit) break;
                    c = *iter;
                }
                value += c;
            }
        }

        SPARQLLexer::Token token = lexer.getNext();
        if (token == SPARQLLexer::At) {
            if (lexer.getNext() != SPARQLLexer::Identifier)
                throw ParserException("language tag expected after '@'");
            subType = Element::CustomLanguage;
            valueType = lexer.getTokenValue();
        } else if (token == SPARQLLexer::Type) {
            token = lexer.getNext();
            if (token == SPARQLLexer::IRI) {
                subType = Element::CustomType;
                valueType = lexer.getIRIValue();
            } else {
                throw ParserException("type URI expeted after '^^'");
            }
        } else {
            lexer.unget(token);
        }
    }
    /// Parse a "IRIrefOrFunction" production
    Filter *parseIRIrefOrFunction(std::map<std::string, unsigned> &localVars, bool mustCall) {
        // The IRI
        if (lexer.getNext() != SPARQLLexer::IRI)
            throw ParserException("IRI expected");
        unique_ptr<Filter> result(new Filter);
        result->type = Filter::IRI;
        result->value = lexer.getIRIValue();

        // Arguments?
        if (lexer.hasNext(SPARQLLexer::LParen)) {
            lexer.getNext();
            unique_ptr<Filter> call(new Filter);
            call->type = Filter::Function;
            call->arg1 = result.release();
            if (lexer.hasNext(SPARQLLexer::RParen)) {
                lexer.getNext();
            } else {
                unique_ptr<Filter> args(new Filter);
                Filter* tail = args.get();
                tail->type = Filter::ArgumentList;
                tail->arg1 = parseExpression(localVars);
                while (true) {
                    if (lexer.hasNext(SPARQLLexer::Comma)) {
                        lexer.getNext();
                        tail = tail->arg2 = new Filter;
                        tail->type = Filter::ArgumentList;
                        tail->arg1 = parseExpression(localVars);
                    } else {
                        if (lexer.getNext() != SPARQLLexer::RParen)
                            throw ParserException("')' expected");
                        break;
                    }
                }
                call->arg2 = args.release();
            }

            result = std::move(call);
        } else if (mustCall) {
            throw ParserException("'(' expected");
        }

        return result.release();
    }
    /// Parse a "BuiltInCall" production
    Filter *parseBuiltInCall(std::map<std::string, unsigned> &localVars) {
        if (lexer.getNext() != SPARQLLexer::Identifier)
            throw ParserException("function name expected");

        unique_ptr<Filter> result(new Filter);
        if (lexer.isKeyword("STR")) {
            result->type = Filter::Builtin_str;
            result->arg1 = parseBrackettedExpression(localVars);
        } else if (lexer.isKeyword("LANG")) {
            result->type = Filter::Builtin_lang;
            result->arg1 = parseBrackettedExpression(localVars);
        } else if (lexer.isKeyword("LANGMATCHES")) {
            result->type = Filter::Builtin_langmatches;
            if (lexer.getNext() != SPARQLLexer::LParen)
                throw ParserException("'(' expected");
            result->arg1 = parseExpression(localVars);
            if (lexer.getNext() != SPARQLLexer::Comma)
                throw ParserException("',' expected");
            result->arg2 = parseExpression(localVars);
            if (lexer.getNext() != SPARQLLexer::RParen)
                throw ParserException("')' expected");
        } else if (lexer.isKeyword("DATATYPE")) {
            result->type = Filter::Builtin_datatype;
            result->arg1 = parseBrackettedExpression(localVars);
        } else if (lexer.isKeyword("BOUND")) {
            result->type = Filter::Builtin_bound;
            if (lexer.getNext() != SPARQLLexer::LParen)
                throw ParserException("'(' expected");
            if (lexer.getNext() != SPARQLLexer::Variable)
                throw ParserException("variable expected as argument to BOUND");
            unique_ptr<Filter> arg(new Filter());
            arg->type = Filter::Variable;
            arg->valueArg = nameVariable(lexer.getTokenValue());
            result->arg1 = arg.release();
            if (lexer.getNext() != SPARQLLexer::RParen)
                throw ParserException("')' expected");
        } else if (lexer.isKeyword("sameTerm")) {
            result->type = Filter::Builtin_sameterm;
            if (lexer.getNext() != SPARQLLexer::LParen)
                throw ParserException("'(' expected");
            result->arg1 = parseExpression(localVars);
            if (lexer.getNext() != SPARQLLexer::Comma)
                throw ParserException("',' expected");
            result->arg2 = parseExpression(localVars);
            if (lexer.getNext() != SPARQLLexer::RParen)
                throw ParserException("')' expected");
        } else if (lexer.isKeyword("isIRI")) {
            result->type = Filter::Builtin_isiri;
            result->arg1 = parseBrackettedExpression(localVars);
        } else if (lexer.isKeyword("isURI")) {
            result->type = Filter::Builtin_isiri;
            result->arg1 = parseBrackettedExpression(localVars);
        } else if (lexer.isKeyword("isBLANK")) {
            result->type = Filter::Builtin_isblank;
            result->arg1 = parseBrackettedExpression(localVars);
        } else if (lexer.isKeyword("isLITERAL")) {
            result->type = Filter::Builtin_isliteral;
            result->arg1 = parseBrackettedExpression(localVars);
        } else if (lexer.isKeyword("REGEX")) {
            result->type = Filter::Builtin_regex;
            if (lexer.getNext() != SPARQLLexer::LParen)
                throw ParserException("'(' expected");
            result->arg1 = parseExpression(localVars);
            if (lexer.getNext() != SPARQLLexer::Comma)
                throw ParserException("',' expected");
            result->arg2 = parseExpression(localVars);
            if (lexer.hasNext(SPARQLLexer::Comma)) {
                lexer.getNext();
                result->arg3 = parseExpression(localVars);
            }
            if (lexer.getNext() != SPARQLLexer::RParen)
                throw ParserException("')' expected");
        } else if (lexer.isKeyword("in")) {
            result->type = Filter::Builtin_in;
            if (lexer.getNext() != SPARQLLexer::LParen)
                throw ParserException("'(' expected");
            result->arg1 = parseExpression(localVars);

            if (lexer.hasNext(SPARQLLexer::RParen)) {
                lexer.getNext();
            } else {
                if (lexer.getNext() != SPARQLLexer::Comma)
                    throw ParserException("',' expected");
                unique_ptr<Filter> args(new Filter);
                Filter* tail = args.get();
                tail->type = Filter::ArgumentList;
                tail->arg1 = parseExpression(localVars);
                while (true) {
                    if (lexer.hasNext(SPARQLLexer::Comma)) {
                        lexer.getNext();
                        tail = tail->arg2 = new Filter;
                        tail->type = Filter::ArgumentList;
                        tail->arg1 = parseExpression(localVars);
                    } else {
                        if (lexer.getNext() != SPARQLLexer::RParen)
                            throw ParserException("')' expected");
                        break;
                    }
                }
                result->arg2 = args.release();
            }
        } else {
            throw ParserException("unknown function '" + lexer.getTokenValue() + "'");
        }
        return result.release();
    }
    /// Parse a "PrimaryExpression" production
    Filter *parsePrimaryExpression(std::map<std::string, unsigned> &localVars) {
        SPARQLLexer::Token token = lexer.getNext();

        if (token == SPARQLLexer::LParen) {
            lexer.unget(token);
            return parseBrackettedExpression(localVars);
        }
        if (token == SPARQLLexer::Identifier) {
            if (lexer.isKeyword("true")) {
                unique_ptr<Filter> result(new Filter);
                result->type = Filter::Literal;
                result->value = "true";
                result->valueType = "http://www.w3.org/2001/XMLSchema#boolean";
                result->valueArg = Element::CustomType;
                return result.release();
            } else if (lexer.isKeyword("false")) {
                unique_ptr<Filter> result(new Filter);
                result->type = Filter::Literal;
                result->value = "false";
                result->valueType = "http://www.w3.org/2001/XMLSchema#boolean";
                result->valueArg = Element::CustomType;
                return result.release();
            }
            lexer.unget(token);
            return parseBuiltInCall(localVars);
        }
        if (token == SPARQLLexer::IRI) {
            lexer.unget(token);
            return parseIRIrefOrFunction(localVars, false);
        }
        if (token == SPARQLLexer::String) {
            lexer.unget(token);
            unique_ptr<Filter> result(new Filter);
            result->type = Filter::Literal;
            Element::SubType type;
            parseRDFLiteral(result->value, type, result->valueType);
            result->valueArg = type;
            return result.release();
        }
        if (token == SPARQLLexer::Integer) {
            unique_ptr<Filter> result(new Filter);
            result->type = Filter::Literal;
            result->value = lexer.getTokenValue();
            result->valueType = "http://www.w3.org/2001/XMLSchema#integer";
            result->valueArg = Element::CustomType;
            return result.release();
        }
        if (token == SPARQLLexer::Decimal) {
            unique_ptr<Filter> result(new Filter);
            result->type = Filter::Literal;
            result->value = lexer.getTokenValue();
            result->valueType = "http://www.w3.org/2001/XMLSchema#decimal";
            result->valueArg = Element::CustomType;
            return result.release();
        }
        if (token == SPARQLLexer::Double) {
            unique_ptr<Filter> result(new Filter);
            result->type = Filter::Literal;
            result->value = lexer.getTokenValue();
            result->valueType = "http://www.w3.org/2001/XMLSchema#double";
            result->valueArg = Element::CustomType;
            return result.release();
        }
        if (token == SPARQLLexer::Variable) {
            unique_ptr<Filter> result(new Filter);
            result->type = Filter::Variable;
            result->value = lexer.getTokenValue();
            result->valueArg = nameVariable(result->value);
            return result.release();
        }
        throw ParserException("syntax error in primary expression");
    }
    /// Parse a "UnaryExpression" production
    Filter *parseUnaryExpression(std::map<std::string, unsigned> &localVars) {
        SPARQLLexer::Token token = lexer.getNext();

        if (token == SPARQLLexer::Not) {
            unique_ptr<Filter> result(new Filter);
            result->type = Filter::Not;
            result->arg1 = parsePrimaryExpression(localVars);
            return result.release();
        } else if (token == SPARQLLexer::Plus) {
            unique_ptr<Filter> result(new Filter);
            result->type = Filter::UnaryPlus;
            result->arg1 = parsePrimaryExpression(localVars);
            return result.release();
        } else if (token == SPARQLLexer::Minus) {
            unique_ptr<Filter> result(new Filter);
            result->type = Filter::UnaryMinus;
            result->arg1 = parsePrimaryExpression(localVars);
            return result.release();
        } else {
            lexer.unget(token);
            return parsePrimaryExpression(localVars);
        }
    }
    /// Parse a "MultiplicativeExpression" production
    Filter *parseMultiplicativeExpression(std::map<std::string, unsigned> &localVars) {
        unique_ptr<Filter> result(parseUnaryExpression(localVars));

        // op *
        while (true) {
            SPARQLLexer::Token token = lexer.getNext();
            if ((token == SPARQLLexer::Mul) || (token == SPARQLLexer::Div)) {
                unique_ptr<Filter> right(parseUnaryExpression(localVars));

                unique_ptr<Filter> newEntry(new Filter);
                switch (token) {
                case SPARQLLexer::Mul: newEntry->type = Filter::Mul; break;
                case SPARQLLexer::Div: newEntry->type = Filter::Div; break;
                default: throw; // cannot happen
                }
                newEntry->arg1 = result.release();
                newEntry->arg2 = right.release();
                result = std::move(newEntry);
            } else {
                lexer.unget(token);
                break;
            }
        }
        return result.release();
    }
    /// Parse a "AdditiveExpression" production
    Filter *parseAdditiveExpression(std::map<std::string, unsigned> &localVars) {
        unique_ptr<Filter> result(parseMultiplicativeExpression(localVars));

        // op *
        while (true) {
            SPARQLLexer::Token token = lexer.getNext();
            if ((token == SPARQLLexer::Plus) || (token == SPARQLLexer::Minus)) {
                unique_ptr<Filter> right(parseMultiplicativeExpression(localVars));

                unique_ptr<Filter> newEntry(new Filter);
                switch (token) {
                case SPARQLLexer::Plus: newEntry->type = Filter::Plus; break;
                case SPARQLLexer::Minus: newEntry->type = Filter::Minus; break;
                default: throw; // cannot happen
                }
                newEntry->arg1 = result.release();
                newEntry->arg2 = right.release();
                result = std::move(newEntry);
            } else {
                lexer.unget(token);
                break;
            }
        }
        return result.release();
    }
    /// Parse a "NumericExpression" production
    Filter *parseNumericExpression(std::map<std::string, unsigned> &localVars) {
        return parseAdditiveExpression(localVars);
    }
    /// Parse a "RelationalExpression" production
    Filter *parseRelationalExpression(std::map<std::string, unsigned> &localVars) {
        unique_ptr<Filter> result(parseNumericExpression(localVars));

        // op *
        while (true) {
            SPARQLLexer::Token token = lexer.getNext();
            if ((token == SPARQLLexer::Equal) || (token == SPARQLLexer::NotEqual) || (token == SPARQLLexer::Less) || (token == SPARQLLexer::LessOrEqual) || (token == SPARQLLexer::Greater) || (token == SPARQLLexer::GreaterOrEqual)) {
                unique_ptr<Filter> right(parseNumericExpression(localVars));

                unique_ptr<Filter> newEntry(new Filter);
                switch (token) {
                case SPARQLLexer::Equal: newEntry->type = Filter::Equal; break;
                case SPARQLLexer::NotEqual: newEntry->type = Filter::NotEqual; break;
                case SPARQLLexer::Less: newEntry->type = Filter::Less; break;
                case SPARQLLexer::LessOrEqual: newEntry->type = Filter::LessOrEqual; break;
                case SPARQLLexer::Greater: newEntry->type = Filter::Greater; break;
                case SPARQLLexer::GreaterOrEqual: newEntry->type = Filter::GreaterOrEqual; break;
                default: throw; // cannot happen
                }
                newEntry->arg1 = result.release();
                newEntry->arg2 = right.release();
                result = std::move(newEntry);
            } else {
                lexer.unget(token);
                break;
            }
        }
        return result.release();
    }
    /// Parse a "ValueLogical" production
    Filter *parseValueLogical(std::map<std::string, unsigned> &localVars) {
        return parseRelationalExpression(localVars);
    }
    /// Parse a "ConditionalAndExpression" production
    Filter *parseConditionalAndExpression(std::map<std::string, unsigned> &localVars) {
        unique_ptr<Filter> result(parseValueLogical(localVars));

        // && *
        while (lexer.hasNext(SPARQLLexer::And)) {
            if (lexer.getNext() != SPARQLLexer::And)
                throw ParserException("'&&' expected");
            unique_ptr<Filter> right(parseValueLogical(localVars));

            unique_ptr<Filter> newEntry(new Filter);
            newEntry->type = Filter::And;
            newEntry->arg1 = result.release();
            newEntry->arg2 = right.release();

            result = std::move(newEntry);
        }
        return result.release();
    }
    /// Parse a "ConditionalOrExpression" production
    Filter *parseConditionalOrExpression(std::map<std::string, unsigned> &localVars) {
        unique_ptr<Filter> result(parseConditionalAndExpression(localVars));

        // || *
        while (lexer.hasNext(SPARQLLexer::Or)) {
            if (lexer.getNext() != SPARQLLexer::Or)
                throw ParserException("'||' expected");
            unique_ptr<Filter> right(parseConditionalAndExpression(localVars));

            unique_ptr<Filter> newEntry(new Filter);
            newEntry->type = Filter::Or;
            newEntry->arg1 = result.release();
            newEntry->arg2 = right.release();

            result = std::move(newEntry);
        }
        return result.release();
    }
    /// Parse a "Expression" production
    Filter *parseExpression(std::map<std::string, unsigned> &localVars) {
        return parseConditionalOrExpression(localVars);
    }
    /// Parse a "BrackettedExpression" production
    Filter *parseBrackettedExpression(std::map<std::string, unsigned> &localVars) {
        // '('
        if (lexer.getNext() != SPARQLLexer::LParen)
            throw ParserException("'(' expected");

        // Expression
        unique_ptr<Filter> result(parseExpression(localVars));

        // ')'
        if (lexer.getNext() != SPARQLLexer::RParen)
            throw ParserException("')' expected");

        return result.release();
    }
    /// Parse a "Constraint" production
    Filter *parseConstraint(std::map<std::string, unsigned> &localVars) {
        // Check possible productions
        if (lexer.hasNext(SPARQLLexer::LParen))
            return parseBrackettedExpression(localVars);
        if (lexer.hasNext(SPARQLLexer::Identifier))
            return parseBuiltInCall(localVars);
        if (lexer.hasNext(SPARQLLexer::IRI))
            return parseIRIrefOrFunction(localVars, true);

        // Report an error
        throw ParserException("filter constraint expected");
    }
    /// Parse a filter condition
    void parseFilter(PatternGroup &group, std::map<std::string, unsigned> &localVars) {
        Filter* entry = parseConstraint(localVars);
        group.filters.push_back(*entry);
        delete entry;
    }
    /// Parse an entry in a pattern
    Element parsePatternElement(PatternGroup&group, std::map<std::string, unsigned> &localVars) {
        Element result;
        SPARQLLexer::Token token = lexer.getNext();
        if (token == SPARQLLexer::Variable) {
            result.type = Element::Variable;
            result.id = nameVariable(lexer.getTokenValue());
        } else if (token == SPARQLLexer::String) {
            result.type = Element::Literal;
            lexer.unget(token);
            parseRDFLiteral(result.value, result.subType, result.subTypeValue);
        } else if (token == SPARQLLexer::IRI) {
            result.type = Element::IRI;
            result.value = lexer.getIRIValue();
        } else if (token == SPARQLLexer::Anon) {
            result.type = Element::Variable;
            result.id = variableCount++;
        } else if (token == SPARQLLexer::LBracket) {
            result = parseBlankNode(group, localVars);
            if (lexer.getNext() != SPARQLLexer::RBracket)
                throw ParserException("']' expected");
        } else if (token == SPARQLLexer::Underscore) {
            // _:variable
            if (lexer.getNext() != SPARQLLexer::Colon)
                throw ParserException("':' expected");
            if (lexer.getNext() != SPARQLLexer::Identifier)
                throw ParserException("identifier expected after ':'");
            result.type = Element::Variable;
            if (localVars.count(lexer.getTokenValue()))
                result.id = localVars[lexer.getTokenValue()]; else
                result.id = localVars[lexer.getTokenValue()] = variableCount++;
        } else if (token == SPARQLLexer::Colon) {
            // :identifier. Should reference the base
            if (lexer.getNext() != SPARQLLexer::Identifier)
                throw ParserException("identifier expected after ':'");
            result.type = Element::IRI;
            result.value = lexer.getTokenValue();
        } else if (token == SPARQLLexer::Identifier) {
            string prefix = lexer.getTokenValue();
            // Handle the keyword 'a'
            if (prefix == "a") {
                result.type = Element::IRI;
                result.value = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type";
            } else {
                // prefix:suffix
                if (lexer.getNext() != SPARQLLexer::Colon)
                    throw ParserException("':' expected after '" + prefix + "'");
                if (!prefixes.count(prefix))
                    throw ParserException("unknown prefix '" + prefix + "'");
                if (lexer.getNext() != SPARQLLexer::Identifier)
                    throw ParserException("identifier expected after ':'");
                result.type = Element::IRI;
                result.value = prefixes[prefix] + lexer.getIRIValue();
            }
        } else if (token == SPARQLLexer::Percent) {
            usingCustomGrammar = true;
            Element predicate = parsePatternElement(group, localVars);
            if (predicate.type != Element::IRI)
                throw ParserException("IRI expected after '%'");
            result.type = Element::Template;
            result.value = predicate.value;
        } else if (token == SPARQLLexer::PREDICATE) {
            usingCustomGrammar = true;
            result.type = Element::Predicate;
        } else {
            throw ParserException("invalid pattern element");
        }
        return result;
    }
    /// Parse blank node patterns
    Element parseBlankNode(PatternGroup &group, std::map<std::string, unsigned> &localVars) {
        // The subject is a blank node
        Element subject;
        subject.type = Element::Variable;
        subject.id = variableCount++;

        // Parse the the remaining part of the pattern
        SPARQLParser::Element predicate = parsePatternElement(group, localVars);
        SPARQLParser::Element object = parsePatternElement(group, localVars);
        group.patterns.push_back(Pattern(subject, predicate, object));

        // Check for the tail
        while (true) {
            SPARQLLexer::Token token = lexer.getNext();
            if (token == SPARQLLexer::Semicolon) {
                predicate = parsePatternElement(group, localVars);
                object = parsePatternElement(group, localVars);
                group.patterns.push_back(Pattern(subject, predicate, object));
                continue;
            } else if (token == SPARQLLexer::Comma) {
                object = parsePatternElement(group, localVars);
                group.patterns.push_back(Pattern(subject, predicate, object));
                continue;
            } else if (token == SPARQLLexer::Dot) {
                return subject;
            } else if (token == SPARQLLexer::RBracket) {
                lexer.unget(token);
                return subject;
            } else if (token == SPARQLLexer::Identifier) {
                if (!lexer.isKeyword("filter"))
                    throw ParserException("'filter' expected");
                parseFilter(group, localVars);
                continue;
            } else {
                // Error while parsing, let out caller handle it
                lexer.unget(token);
                return subject;
            }
        }
    }
    // Parse a graph pattern
    void parseGraphPattern(PatternGroup &group) {
        map<string, unsigned> localVars;

        // Parse the first pattern
        Element subject = parsePatternElement(group, localVars);
        Element predicate = parsePatternElement(group, localVars);
        Element object = parsePatternElement(group, localVars);
        group.patterns.push_back(Pattern(subject, predicate, object));
        // Check for the tail
        while (true) {
            SPARQLLexer::Token token = lexer.getNext();
            if (token == SPARQLLexer::Semicolon) {
                predicate = parsePatternElement(group, localVars);
                object = parsePatternElement(group, localVars);
                group.patterns.push_back(Pattern(subject, predicate, object));
                continue;
            } else if (token == SPARQLLexer::Comma) {
                object = parsePatternElement(group, localVars);
                group.patterns.push_back(Pattern(subject, predicate, object));
                continue;
            } else if (token == SPARQLLexer::Dot) {
                return;
            } else if (token == SPARQLLexer::LArrow) {
                usingCustomGrammar = true;
                Pattern last_pattern = group.patterns.back();
                Pattern pattern(last_pattern.object, last_pattern.predicate, last_pattern.subject);
                pattern.direction = IN;
                group.patterns.pop_back();
                group.patterns.push_back(pattern);
                return;
            } else if (token == SPARQLLexer::RArrow) {
                usingCustomGrammar = true;
                return;
            } else if (token == SPARQLLexer::RCurly) {
                lexer.unget(token);
                return;
            } else if (token == SPARQLLexer::Identifier) {
                if (!lexer.isKeyword("filter"))
                    throw ParserException("'filter' expected");
                parseFilter(group, localVars);
                continue;
            } else {
                // Error while parsing, let our caller handle it
                lexer.unget(token);
                return;
            }
        }
    }
    // Parse a group of patterns
    void parseGroupGraphPattern(PatternGroup &group) {
        while (true) {
            SPARQLLexer::Token token = lexer.getNext();

            if (token == SPARQLLexer::LCurly) {
                // Parse the group
                PatternGroup newGroup;
                parseGroupGraphPattern(newGroup);

                // Union statement?
                token = lexer.getNext();
                if ((token == SPARQLLexer::Identifier) && (lexer.isKeyword("union"))) {
                    group.unions.push_back(newGroup);
                    while (true) {
                        if (lexer.getNext() != SPARQLLexer::LCurly)
                            throw ParserException("'{' expected");
                        PatternGroup subGroup;
                        parseGroupGraphPattern(subGroup);
                        group.unions.push_back(subGroup);

                        // Another union?
                        token = lexer.getNext();
                        if ((token == SPARQLLexer::Identifier) && (lexer.isKeyword("union")))
                            continue;
                        break;
                    }
                } else {
                    // No, simply merge it
                    group.patterns.insert(group.patterns.end(), newGroup.patterns.begin(), newGroup.patterns.end());
                    group.filters.insert(group.filters.end(), newGroup.filters.begin(), newGroup.filters.end());
                    group.optional.insert(group.optional.end(), newGroup.optional.begin(), newGroup.optional.end());
                    group.unions.insert(group.unions.end(), newGroup.unions.begin(), newGroup.unions.end());
                }
                if (token != SPARQLLexer::Dot)
                    lexer.unget(token);
            } else if ((token == SPARQLLexer::IRI) || (token == SPARQLLexer::Variable)
                       || (token == SPARQLLexer::Identifier) || (token == SPARQLLexer::String)
                       || (token == SPARQLLexer::Underscore) || (token == SPARQLLexer::Colon)
                       || (token == SPARQLLexer::LBracket) || (token == SPARQLLexer::Anon)
                       || (token == SPARQLLexer::Percent)) {
                // Distinguish filter conditions
                if ((token == SPARQLLexer::Identifier) && (lexer.isKeyword("filter"))) {
                    map<string, unsigned> localVars;
                    parseFilter(group, localVars);
                } else if ((token == SPARQLLexer::Identifier) && (lexer.isKeyword("optional"))) {
                    if (lexer.getNext() != SPARQLLexer::LCurly)
                        throw ParserException("'{' expected");
                    PatternGroup new_group;
                    parseGroupGraphPattern(new_group);
                    group.optional.push_back(new_group);
                } else {
                    lexer.unget(token);
                    parseGraphPattern(group);
                }
            } else if (token == SPARQLLexer::RCurly) {
                break;
            } else if (token == SPARQLLexer::Dot) {
                continue;
            } else {
                throw ParserException("'}' expected");
            }
        }
    }

    /// Parse the prefix part if any
    void parsePrefix() {
        while (true) {
            SPARQLLexer::Token token = lexer.getNext();

            if ((token == SPARQLLexer::Identifier) && (lexer.isKeyword("prefix"))) {
                // Parse the prefix entry
                if (lexer.getNext() != SPARQLLexer::Identifier)
                    throw ParserException("prefix name expected");
                string name = lexer.getTokenValue();
                if (lexer.getNext() != SPARQLLexer::Colon)
                    throw ParserException("':' expected");
                if (lexer.getNext() != SPARQLLexer::IRI)
                    throw ParserException("IRI expected");
                string iri = lexer.getIRIValue();

                // Register the new prefix
                if (prefixes.count(name))
                    throw ParserException("duplicate prefix '" + name + "'");
                prefixes[name] = iri;
            } else {
                lexer.unget(token);
                return;
            }
        }
    }
    /// Parse corun
    void parseCorun() {
        while (true) {
            SPARQLLexer::Token token = lexer.getNext();
            if ((token == SPARQLLexer::Identifier) && (lexer.isKeyword("corun"))) {
                usingCustomGrammar = true;
                // Parse the corun entry
                if (lexer.getNext() != SPARQLLexer::Integer)
                    throw ParserException("Integer(corun step) expected");
                string corun_step_str = lexer.getTokenValue();
                corun_step = stoi(corun_step_str);
                if (lexer.getNext() != SPARQLLexer::Integer)
                    throw ParserException("Integer(fetch step) expected");
                string fetch_step_str = lexer.getTokenValue();
                fetch_step = stoi(fetch_step_str);
                corun_enabled = true;
                /// FIXME: more than one CORUN/FETCH
            } else {
                lexer.unget(token);
                return;
            }
        }
    }
    /// Parse the projection
    void parseProjection() {
        // Parse the projection
        if ((lexer.getNext() != SPARQLLexer::Identifier) || (!lexer.isKeyword("select")))
            throw ParserException("'select' expected");

        // Parse modifiers, if any
        {
            SPARQLLexer::Token token = lexer.getNext();
            if (token == SPARQLLexer::Identifier) {
                if (lexer.isKeyword("distinct")) projectionModifier = Modifier_Distinct;
                else if (lexer.isKeyword("reduced")) projectionModifier = Modifier_Reduced;
                else if (lexer.isKeyword("count")) projectionModifier = Modifier_Count;
                else if (lexer.isKeyword("duplicates")) projectionModifier = Modifier_Duplicates;
                else lexer.unget(token);
            } else lexer.unget(token);
        }

        // Parse the projection clause
        bool first = true;
        while (true) {
            SPARQLLexer::Token token = lexer.getNext();
            if (token == SPARQLLexer::Variable) {
                projection.push_back(nameVariable(lexer.getTokenValue()));
            } else if (token == SPARQLLexer::Mul) {
                // We do nothing here. Empty projections will be filled with all
                // named variables after parsing
            } else {
                if (first)
                    throw ParserException("projection required after select");
                lexer.unget(token);
                break;
            }
            first = false;
        }
    }
    /// Parse the from part if any
    void parseFrom() {
        while (true) {
            SPARQLLexer::Token token = lexer.getNext();

            if ((token == SPARQLLexer::Identifier) && (lexer.isKeyword("from"))) {
                throw ParserException("from clause currently not implemented");
            } else {
                lexer.unget(token);
                return;
            }
        }
    }
    /// Parse the where part if any
    void parseWhere() {
        if ((lexer.getNext() != SPARQLLexer::Identifier) || (!lexer.isKeyword("where")))
            throw ParserException("'where' expected");
        if (lexer.getNext() != SPARQLLexer::LCurly)
            throw ParserException("'{' expected");

        patterns = PatternGroup();
        parseGroupGraphPattern(patterns);
    }
    /// Parse the order by part if any
    void parseOrderBy() {
        SPARQLLexer::Token token = lexer.getNext();
        if ((token != SPARQLLexer::Identifier) || (!lexer.isKeyword("order"))) {
            lexer.unget(token);
            return;
        }
        if ((lexer.getNext() != SPARQLLexer::Identifier) || (!lexer.isKeyword("by")))
            throw ParserException("'by' expected");

        while (true) {
            token = lexer.getNext();
            if (token == SPARQLLexer::Identifier) {
                if (lexer.isKeyword("asc") || lexer.isKeyword("desc")) {
                    Order o;
                    o.descending = lexer.isKeyword("desc");
                    if (lexer.getNext() != SPARQLLexer::LParen)
                        throw ParserException("'(' expected");
                    token = lexer.getNext();
                    if ((token == SPARQLLexer::Identifier) && (lexer.isKeyword("count"))) {
                        o.id = ~0u;
                    } else if (token == SPARQLLexer::Variable) {
                        o.id = nameVariable(lexer.getTokenValue());
                    } else throw ParserException("variable expected in order-by clause");
                    if (lexer.getNext() != SPARQLLexer::RParen)
                        throw ParserException("')' expected");
                    order.push_back(o);
                } else if (lexer.isKeyword("count")) {
                    Order o; o.id = ~0u; o.descending = false;
                    order.push_back(o);
                } else {
                    lexer.unget(token);
                    return;
                }
            } else if (token == SPARQLLexer::Variable) {
                Order o;
                o.id = nameVariable(lexer.getTokenValue());
                o.descending = false;
                order.push_back(o);
            } else if (token == SPARQLLexer::Eof) {
                lexer.unget(token);
                return;
            } else {
                throw ParserException("variable expected in order-by clause");
            }
        }
    }
    /// Parse the limit part if any
    void parseLimit() {
        SPARQLLexer::Token token = lexer.getNext();

        if ((token == SPARQLLexer::Identifier) && (lexer.isKeyword("limit"))) {
            if (lexer.getNext() != SPARQLLexer::Integer)
                throw ParserException("number expected after 'limit'");
            limit = atoi(lexer.getTokenValue().c_str());
            if (limit < 0)
                throw ParserException("invalid limit specifier");
        } else {
            lexer.unget(token);
        }
    }
    /// Parse the offset part if any
    void parseOffset() {
        SPARQLLexer::Token token = lexer.getNext();

        if ((token == SPARQLLexer::Identifier) && (lexer.isKeyword("offset"))) {
            if (lexer.getNext() != SPARQLLexer::Integer)
                throw ParserException("number expected after 'offset'");
            offset = atoi(lexer.getTokenValue().c_str());
            if (offset < 0)
                throw ParserException("invalid offset specifier");
        } else {
            lexer.unget(token);
        }
    }

public:
    /// Constructor
    explicit SPARQLParser(SPARQLLexer &lexer)
        : lexer(lexer), variableCount(0), namedVariableCount(0),
          projectionModifier(Modifier_None), limit(-1), offset(0u) {
        usingCustomGrammar = false;
        corun_enabled = false;
        corun_step = 0;
        fetch_step = 0;
    }
    /// Destructor
    ~SPARQLParser() { }

    /// Parse the input. Throws an exception in the case of an error
    void parse(bool multiQuery = false) {
        // Parse the prefix part
        parsePrefix();

        // Parse the corun part
        parseCorun();

        // Parse the projection
        parseProjection();

        // Parse the from clause
        parseFrom();

        // Parse the where clause
        parseWhere();

        // Parse the order by clause
        parseOrderBy();

        // Parse the limit clause
        parseLimit();

        // Parse the offset clause
        parseOffset();

        // Check that the input is done
        if ((!multiQuery) && (lexer.getNext() != SPARQLLexer::Eof))
            throw ParserException("syntax error");

        // Fixup empty projections (i.e. *)
        if (!projection.size()) {
            for (map<string, int>::const_iterator iter = namedVariables.begin(), limit = namedVariables.end();
                    iter != limit; ++iter)
                projection.push_back((*iter).second);
        }
    }

    /// Get the patterns
    const PatternGroup &getPatterns() const { return patterns; }
    /// Get the name of a variable
    std::string getVariableName(int id) const {
        for (map<string, int>::const_iterator iter = namedVariables.begin(), limit = namedVariables.end();
                iter != limit; ++iter)
            if ((*iter).second == id)
                return (*iter).first;
        return "";
    }

    /// Iterator over the projection clause
    typedef std::vector<int>::const_iterator projection_iterator;
    /// Iterator over the projection
    projection_iterator projectionBegin() const { return projection.begin(); }
    /// Iterator over the projection
    projection_iterator projectionEnd() const { return projection.end(); }

    /// Iterator over the order by clause
    typedef std::vector<Order>::const_iterator order_iterator;
    /// Iterator over the order by clause
    order_iterator orderBegin() const { return order.begin(); }
    /// Iterator over the order by clause
    order_iterator orderEnd() const { return order.end(); }

    /// The projection modifier
    ProjectionModifier getProjectionModifier() const { return projectionModifier; }
    /// The size limit
    int getLimit() const { return limit; }
    /// The offset
    unsigned getOffset() const { return offset; }
    /// Get the variableCount
    unsigned getVariableCount() const { return variableCount; }
    // indicate if custom grammar is in use
    bool isUsingCustomGrammar() const { return usingCustomGrammar; }
    // indicate if corun optimization is in use
    bool isCorunEnabled() const { return corun_enabled; }
    // get the corun step
    int getCorunStep() const { return corun_step; }
    // get the fetch step
    int getFetchStep() const { return fetch_step; }
};
