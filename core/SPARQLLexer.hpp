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

#include <cstring> //include header for strncmp
#include <string>

using namespace std;

/// A lexer for SPARQL input
class SPARQLLexer {
public:
    /// Possible tokens
    enum Token { None, Error, Eof, IRI, String, Variable, Identifier, Colon,
                 Semicolon, Comma, Dot, Underscore, LCurly, RCurly, LParen,
                 RParen, LBracket, RBracket, LArrow, RArrow, Anon, Equal,
                 NotEqual, Less, LessOrEqual, Greater, GreaterOrEqual, At,
                 Type, Not, Or, And, Plus, Minus, Mul, Div, Integer, Decimal,
                 Double, Percent, PREDICATE
               };

private:
    /// The input
    std::string input;
    /// The current position
    std::string::const_iterator pos;
    /// The start of the current token
    std::string::const_iterator tokenStart;
    /// The end of the curent token. Only set if delimiters are stripped
    std::string::const_iterator tokenEnd;
    /// The token put back with unget
    Token putBack;
    /// Was the doken end set?
    bool hasTokenEnd;

public:
    /// Constructor
    SPARQLLexer(const std::string &input)
        : input(input), pos(this->input.begin()),
          tokenStart(pos), tokenEnd(pos),
          putBack(None), hasTokenEnd(false) { }

    /// Destructor
    ~SPARQLLexer() { }

    /// Get the next token
    Token getNext() {
        // Do we have a token already?
        if (putBack != None) {
            Token result = putBack;
            putBack = None;
            return result;
        }

        // Reset the token end
        hasTokenEnd = false;

        // Read the string
        while (pos != input.end()) {
            tokenStart = pos;
            // Interpret the first character
            switch (*(pos++)) {
            // Whitespace
            case ' ': case '\t': case '\n': case '\r': case '\f': continue;
            // Single line comment
            case '#':
                while (pos != input.end()) {
                    if (((*pos) == '\n') || ((*pos) == '\r'))
                        break;
                    ++pos;
                }
                if (pos != input.end()) ++pos;
                continue;
            // Simple tokens
            case ':': return Colon;
            case ';': return Semicolon;
            case ',': return Comma;
            case '.': return Dot;
            case '_':
                if (input.end() - pos >= 12
                        && strncmp(&*pos, "_PREDICATE__", 12) == 0) {
                    pos += 12;
                    return PREDICATE;
                }
                return Underscore;
            case '{': return LCurly;
            case '}': return RCurly;
            case '(': return LParen;
            case ')': return RParen;
            case '@': return At;
            case '+': return Plus;
            case '-':
                // check if it is a right arrow
                if ((pos != input.end()) && ((*pos) == '>')) {
                    pos++;
                    return RArrow;
                }
                return Minus;
            case '*': return Mul;
            case '/': return Div;
            case '=': return Equal;
            case '%': return Percent;
            // Not equal
            case '!':
                if ((pos == input.end()) || ((*pos) != '='))
                    return Not;
                ++pos;
                return NotEqual;
            // Brackets
            case '[':
                // Skip whitespaces
                while (pos != input.end()) {
                    switch (*pos) {
                    case ' ': case '\t': case '\n': case '\r': case '\f':
                        ++pos;
                        continue;
                    }
                    break;
                }
                // Check for a closing ]
                if ((pos != input.end()) && ((*pos) == ']')) {
                    ++pos;
                    return Anon;
                }
                return LBracket;
            case ']': return RBracket;
            // Greater
            case '>':
                if ((pos != input.end()) && ((*pos) == '=')) {
                    ++pos;
                    return GreaterOrEqual;
                }
                return Greater;
            // Type
            case '^':
                if ((pos == input.end()) || ((*pos) != '^'))
                    return Error;
                ++pos;
                return Type;
            // Or
            case '|':
                if ((pos == input.end()) || ((*pos) != '|'))
                    return Error;
                ++pos;
                return Or;
            // And
            case '&':
                if ((pos == input.end()) || ((*pos) != '&'))
                    return Error;
                ++pos;
                return And;
            // IRI Ref
            case '<':
                tokenStart = pos;
                // Try to parse as URI
                for (; pos != input.end(); ++pos) {
                    char c = *pos;
                    // Escape chars
                    if (c == '\\') {
                        if ((++pos) == input.end()) break;
                        continue;
                    }
                    // Fast tests
                    if ((c >= 'a') && (c <= 'z')) continue;
                    if ((c >= 'A') && (c <= 'Z')) continue;

                    // Test for invalid characters
                    if ((c == '<') || (c == '>')
                            || (c == '\"') || (c == '`')
                            || (c == '{') || (c == '}')
                            || (c == '^') || (c == '|')
                            || ((c & 0xFF) <= 0x20))
                        break;
                }

                // Successful parse?
                if ((pos != input.end()) && ((*pos) == '>')) {
                    tokenEnd = pos; hasTokenEnd = true;
                    ++pos;
                    return IRI;
                }
                pos = tokenStart;

                // No, do we have a less-or-equal?
                if ((pos != input.end()) && ((*pos) == '=')) {
                    pos++;
                    return LessOrEqual;
                }
                // No, do we have a left arrow?
                if ((pos != input.end()) && ((*pos) == '-')) {
                    pos++;
                    return LArrow;
                }
                // Just a less
                return Less;
            // String
            case '\'':
                tokenStart = pos;
                while (pos != input.end()) {
                    if ((*pos) == '\\') {
                        ++pos;
                        if (pos != input.end()) ++pos;
                        continue;
                    }
                    if ((*pos) == '\'')
                        break;
                    ++pos;
                }
                tokenEnd = pos; hasTokenEnd = true;
                if (pos != input.end()) ++pos;
                return String;
            // String
            case '\"':
                tokenStart = pos;
                while (pos != input.end()) {
                    if ((*pos) == '\\') {
                        ++pos;
                        if (pos != input.end()) ++pos;
                        continue;
                    }
                    if ((*pos) == '\"')
                        break;
                    ++pos;
                }
                tokenEnd = pos; hasTokenEnd = true;
                if (pos != input.end()) ++pos;
                return String;
            // Variables
            case '?': case '$':
                tokenStart = pos;
                while (pos != input.end()) {
                    char c = *pos;
                    if (((c >= '0') && (c <= '9'))
                            || ((c >= 'A') && (c <= 'Z'))
                            || ((c >= 'a') && (c <= 'z'))) {
                        ++pos;
                    } else break;
                }
                tokenEnd = pos; hasTokenEnd = true;
                return Variable;
            // Number
            case '0': case '1': case '2': case '3': case '4':
            case '5': case '6': case '7': case '8': case '9':
                while (pos != input.end()) {
                    char c = *pos;
                    if ((c >= '0') && (c <= '9')) {
                        ++pos;
                    } else break;
                }
                tokenEnd = pos; hasTokenEnd = true;
                return Integer;
            // Identifier
            default:
                --pos;
                while (pos != input.end()) {
                    char c = *pos;
                    if (((c >= '0') && (c <= '9'))
                            || ((c >= 'A') && (c <= 'Z'))
                            || ((c >= 'a') && (c <= 'z')))
                        ++pos;
                    else
                        break;
                }

                if (pos == tokenStart)
                    return Error;
                return Identifier;
            }
        }
        return Eof;
    }

    /// Get the value of the current token
    std::string getTokenValue() const {
        if (hasTokenEnd)
            return std::string(tokenStart, tokenEnd); else
            return std::string(tokenStart, pos);
    }

    /// Get the value of the current token interpreting IRI escapes
    std::string getIRIValue() const {
        std::string::const_iterator limit = (hasTokenEnd ? tokenEnd : pos);
        std::string result;
        for (std::string::const_iterator iter = tokenStart,
                limit = (hasTokenEnd ? tokenEnd : pos);
                iter != limit; ++iter) {
            char c = *iter;
            if (c == '\\') {
                if ((++iter) == limit) break;
                c = *iter;
            }
            result += c;
        }
        return result;
    }

    /// Get the value of the current token interpreting literal escapes
    std::string getLiteralValue() const {
        std::string::const_iterator limit = (hasTokenEnd ? tokenEnd : pos);
        std::string result;
        for (std::string::const_iterator iter = tokenStart,
                limit = (hasTokenEnd ? tokenEnd : pos);
                iter != limit; ++iter) {
            char c = *iter;
            if (c == '\\') {
                if ((++iter) == limit) break;
                c = *iter;
            }
            result += c;
        }
        return result;
    }

    /// Check if the current token matches a keyword
    bool isKeyword(const char* keyword) const {
        std::string::const_iterator iter = tokenStart,
                                    limit = hasTokenEnd ? tokenEnd : pos;

        while (iter != limit) {
            char c = *iter;
            if ((c >= 'A') && (c <= 'Z')) c += 'a' - 'A';
            char c2 = *keyword;
            if ((c2 >= 'A') && (c2 <= 'Z')) c2 += 'a' - 'A';
            if (c != c2)
                return false;
            if (!*keyword) return false;
            ++iter; ++keyword;
        }
        return !*keyword;
    }

    /// Put the last token back
    void unget(Token value) { putBack = value; }

    /// Peek at the next token
    bool hasNext(Token value) {
        Token peek = getNext();
        unget(peek);
        return peek == value;
    }

    /// Return the read pointer
    std::string::const_iterator getReader() const {
        return (putBack != None) ? tokenStart : pos;
    }
};
