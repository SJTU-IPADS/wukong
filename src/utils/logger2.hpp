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

/*
 * Copyright (c) 2009 Carnegie Mellon University.
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
 *      http://www.graphlab.ml.cmu.edu
 *
 */

#pragma once

#include <pthread.h>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
// for va_start/end
#include <cstdarg>

#include <iostream>

// if set, all log will contain their FILE LINE FUNCTION info
// #define PRINTFILEINFO

// If set, logs to screen will be printed in color
#define COLOROUTPUT

/* control the printout color */
#define RESET 0
#define BRIGHT 1
#define DIM 2
#define UNDERLINE 3
#define BLINK 4
#define REVERSE 7
#define HIDDEN 8

#define BLACK 0
#define RED 1
#define GREEN 2
#define YELLOW 3
#define BLUE 4
#define MAGENTA 5
#define CYAN 6
#define WHITE 7

void textcolor(FILE *handle, int attr, int fg) {
    char command[13];
    /* Command is the control command to the terminal */
    sprintf(command, "%c[%d;%dm", 0x1B, attr, fg + 30);
    fprintf(handle, "%s", command);
}

void reset_color(FILE *handle) {
    char command[20];
    /* Command is the control command to the terminal */
    sprintf(command, "%c[0m", 0x1B);
    fprintf(handle, "%s", command);
}

/**
 * \def LOG_FATAL
 *   Used for fatal and probably irrecoverable conditions
 * \def LOG_ERROR
 *   Used for errors which are recoverable within the scope of the function
 * \def LOG_WARNING
 *   Logs interesting conditions which are probably not fatal
 * \def LOG_EMPH
 *   Outputs as LOG_INFO, but in LOG_WARNING colors. Useful for
 *   outputting information you want to emphasize.
 * \def LOG_INFO
 *   Used for providing general useful information
 * \def LOG_DEBUG
 *   Debugging purposes only
 * \def LOG_EVERYTHING
 *   Log everything
 */
#define LOG_NONE 7
#define LOG_FATAL 6
#define LOG_ERROR 5
#define LOG_WARNING 4
#define LOG_EMPH 3
#define LOG_INFO 2
#define LOG_DEBUG 1
#define LOG_EVERYTHING 0

const char *levelname[] = {
    "EVERYTHING", "DEBUG", "INFO", "EMPH",
    "WARNING", "ERROR", "FATAL", "NONE"
};

const char *prefixes[] = {
    "DEBUG:    ", "DEBUG:    ", "INFO:     ", "INFO:     ",
    "WARNING:  ", "ERROR:    ", "FATAL:    ", ""
};

#ifndef OUTPUTLEVEL
#define OUTPUTLEVEL LOG_DEBUG
#endif

// logger_impl for every single thread(get\set by the get\setSpecific function)
namespace logger_impl {
struct streambuf_entry {
    std::stringstream streambuffer;
    bool streamactive;
};
}  // namespace logger_impl

void streambuffdestructor(void *v) {
    logger_impl::streambuf_entry *t =
        reinterpret_cast<logger_impl::streambuf_entry *>(v);
    delete t;
}

/*
file_logger.
This class control the log being logged to log_file and/or console
*/
class file_logger {
private:
    std::ofstream fout;
    std::string log_file;

    pthread_key_t streambufkey;

    int streamloglevel;
    pthread_mutex_t mut;

    bool log_to_console;
    int log_level;

public:
    file_logger() {
        log_file = "";
        log_to_console = true;
        log_level = LOG_INFO;
        pthread_mutex_init(&mut, NULL);
        pthread_key_create(&streambufkey, streambuffdestructor);
    }

    ~file_logger() {
        if (fout.good()) {
            fout.flush();
            fout.close();
        }

        pthread_mutex_destroy(&mut);
    }

    bool set_log_file(std::string file) {
        // close the file if it is open
        if (fout.good()) {
            fout.flush();
            fout.close();
            log_file = "";
        }

        // if file != "", then open a new file
        if (file.length() > 0) {
            fout.open(file.c_str());
            if (fout.fail()) return false;
            log_file = file;
        }
        return true;
    }

    // Return the current logger file name.
    std::string get_log_file(void) { return log_file; }

    // If consolelog is true, subsequent logger output will be written to stderr
    void set_log_to_console(bool consolelog) { log_to_console = consolelog; }

    // Return true if output is being written to stderr
    bool get_log_to_console() { return log_to_console; }

    // Set the current logger level. All logging commands below the current
    // logger level will not be written.
    void set_log_level(int new_log_level) { log_level = new_log_level; }

    // Return the current logger level
    int get_log_level() { return log_level; }

    // operators
    template <typename T>
    file_logger &operator<<(T a) {
        // get the stream buffer entry of specific thread first
        logger_impl::streambuf_entry *streambufentry =
            reinterpret_cast<logger_impl::streambuf_entry *>(
                pthread_getspecific(streambufkey));

        if (streambufentry != NULL) {
            std::stringstream &sstream = streambufentry->streambuffer;
            bool &streamactive = streambufentry->streamactive;

            if (streamactive)
                sstream << a;
        }
        return *this;
    }

    // if input a std::endl then flush the message to console and/or fout
    file_logger &operator<<(std::ostream & (*f)(std::ostream &)) {
        // get the stream buffer entry first
        logger_impl::streambuf_entry *streambufentry =
            reinterpret_cast<logger_impl::streambuf_entry *>(
                pthread_getspecific(streambufkey));

        if (streambufentry != NULL) {
            std::stringstream &sstream = streambufentry->streambuffer;
            bool &streamactive = streambufentry->streamactive;

            // check whether the input is std::endl;
            typedef std::ostream &(*endltype)(std::ostream &);
            if (streamactive) {
                if (endltype(f) == endltype(std::endl)) {
                    sstream << "\n";
                    stream_flush();
                }
            }
        }
        return *this;
    }

    // F-file, C-console
    void _print2FC(int loglevel, const char *buf, int len) {
        if (fout.good()) {
            pthread_mutex_lock(&mut);
            fout.write(buf, len);
            pthread_mutex_unlock(&mut);
        }

        if (log_to_console) {
#ifdef COLOROUTPUT
            pthread_mutex_lock(&mut);

            // set color
            if (loglevel == LOG_FATAL)
                textcolor(stdout, BRIGHT, RED);
            else if (loglevel == LOG_ERROR)
                textcolor(stdout, BRIGHT, RED);
            else if (loglevel == LOG_WARNING)
                textcolor(stdout, BRIGHT, MAGENTA);
            else if (loglevel == LOG_DEBUG)
                textcolor(stdout, BRIGHT, YELLOW);
            else if (loglevel == LOG_EMPH)
                textcolor(stdout, BRIGHT, GREEN);
#endif
            // in case conflict with cout
            // std::cerr.write(buf, len);
            std::cout.write(buf, len);
#ifdef COLOROUTPUT
            reset_color(stdout);
            pthread_mutex_unlock(&mut);
#endif
        }
    }

    void stream_flush() {
        // get the stream buffer entry first
        logger_impl::streambuf_entry *streambufentry =
            reinterpret_cast<logger_impl::streambuf_entry *>(
                pthread_getspecific(streambufkey));
        if (streambufentry != NULL) {
            std::stringstream &sstream = streambufentry->streambuffer;

            // streambuffer.flush();
            _print2FC(streamloglevel, sstream.str().c_str(),
                      (int)(sstream.str().length()));
            sstream.str("");
        }
    }

    // if the end is "\n" then flush the message to console and/or fout
    file_logger &operator<<(const char *a) {
        // get the stream buffer entry first
        logger_impl::streambuf_entry *streambufentry =
            reinterpret_cast<logger_impl::streambuf_entry *>(
                pthread_getspecific(streambufkey));

        if (streambufentry != NULL) {
            std::stringstream &sstream = streambufentry->streambuffer;
            bool &streamactive = streambufentry->streamactive;

            if (streamactive) {
                sstream << a;
                if (a[strlen(a) - 1] == '\n')
                    stream_flush();
            }
        }
        return *this;
    }

    /**
     * use the stream operator to log
     *
     * lineloglevel: the log's loglevel(if greater than log_level then log)
     * file:  File where the logger call originated
     * function: Function where the logger call originated
     * line: line Line number where the logger call originated
     * do_start: decide the true/false of streamactive(the streamactive decide
     * whether the stream operator being used to print the log to pthread-specific
     * stringstream buffer)
     */
    file_logger &start_stream(int lineloglevel, const char *file,
                              const char *function, int line,
                              bool do_start = true) {
        // get the pthread-specific stream buffer
        logger_impl::streambuf_entry *streambufentry =
            reinterpret_cast<logger_impl::streambuf_entry *>(
                pthread_getspecific(streambufkey));
        // create the key if it doesn't exist
        if (streambufentry == NULL) {
            streambufentry = new logger_impl::streambuf_entry;
            pthread_setspecific(streambufkey, streambufentry);
        }

        std::stringstream &streambuffer = streambufentry->streambuffer;
        bool &streamactive = streambufentry->streamactive;

        if (lineloglevel >= log_level) {
            if (do_start == false) {
                streamactive = false;
                return *this;
            }

            // char *strchr(const char* _Str,char _Val)--to find the location
            // firstly matched _Val in _Str
            file = ((strrchr(file, '/') ? : file - 1) + 1);

            // print header to the streambuffer
            if (streambuffer.str().length() == 0) {
#ifndef PRINTFILEINFO
                streambuffer << prefixes[lineloglevel];
                if (lineloglevel == LOG_DEBUG)
                    streambuffer << file << "(" << function << ":" << line << "):";
#else
                streambuffer << prefixes[lineloglevel] << file << "(" << function << ":"
                             << line << "):";
#endif
            }
            streamactive = true;
            streamloglevel = lineloglevel;
        } else {
            streamactive = false;
        }
        // return the file_logger itself
        return *this;
    }

    /**
     * logs the message if loglevel>=OUTPUTLEVEL
     *
     * loglevel: the log's loglevel(if greater than log_level then log)
     * file:  File where the logger call originated
     * function: Function where the logger call originated
     * line: line Line number where the logger call originated
     * fmt: printf format string
     * arg: variable args. The args will be print in fmt
     */
    void _log(int loglevel, const char *file, const char *function, int line,
              const char *fmt, va_list arg) {
        // check the loglevel
        if (loglevel >= log_level) {
            // the +1 at the end is to recover the file=(-1) which means no-match to 0
            // which means the head
            file = (strchr(file, '/') ? : file - 1) + 1;

            char str[1024];
            int byteswritten;

#ifndef PRINTFILEINFO
            // print loglevel
            if (loglevel == LOG_DEBUG)
                byteswritten = snprintf(str, 1024, "%s%s(%s:%d): ", prefixes[loglevel],
                                        file, function, line);
            else
                byteswritten = snprintf(str, 1024, "%s", prefixes[loglevel]);
#else
            // the actual header
            byteswritten = snprintf(str, 1024, "%s%s(%s:%d): ", prefixes[loglevel],
                                    file, function, line);
#endif
            // the actual logger
            byteswritten +=
                vsnprintf(str + byteswritten, 1024 - byteswritten, fmt, arg);

            // the logger tail
            str[byteswritten] = '\n';
            str[byteswritten + 1] = 0;

            if (fout.good()) {
                pthread_mutex_lock(&mut);
                fout << str;
                pthread_mutex_unlock(&mut);
            }

            // print to the console
            if (log_to_console) {
#ifdef COLOROUTPUT
                pthread_mutex_lock(&mut);
                if (loglevel == LOG_FATAL)
                    textcolor(stdout, BRIGHT, RED);
                else if (loglevel == LOG_ERROR)
                    textcolor(stdout, BRIGHT, RED);
                else if (loglevel == LOG_WARNING)
                    textcolor(stdout, BRIGHT, MAGENTA);
                else if (loglevel == LOG_DEBUG)
                    textcolor(stdout, BRIGHT, YELLOW);
                else if (loglevel == LOG_EMPH)
                    textcolor(stdout, BRIGHT, GREEN);
#endif
                // in case conflict with cout
                // std::cerr << str;
                std::cout << str;
#ifdef COLOROUTPUT
                reset_color(stdout);
                pthread_mutex_unlock(&mut);
#endif
            }
        }
    }
};

file_logger &global_logger() {
    static file_logger l;
    return l;
}

/**
Wrapper to generate 0 code if the output level is lower than the log level
*/
template <bool dostuff>
struct log_dispatch {};

template <>
struct log_dispatch<true> {
    inline static void exec(int loglevel, const char *file, const char *function,
                            int line, const char *fmt, ...) {
        va_list argp;
        va_start(argp, fmt);
        global_logger()._log(loglevel, file, function, line, fmt, argp);
        va_end(argp);
    }
};

template <>
struct log_dispatch<false> {
    inline static void exec(int loglevel, const char *file, const char *function,
                            int line, const char *fmt, ...) {}
};

struct null_stream {
    template <typename T>
    inline null_stream operator<<(T t) {
        return null_stream();
    }
    inline null_stream operator<<(const char *a) { return null_stream(); }
    inline null_stream operator<<(std::ostream & (*f)(std::ostream &)) {
        return null_stream();
    }
};

template <bool dostuff>
struct log_stream_dispatch { };

template <>
struct log_stream_dispatch<true> {
    inline static file_logger &exec(int lineloglevel, const char *file,
                                    const char *function, int line,
                                    bool do_start = true) {
        return global_logger().start_stream(lineloglevel, file, function, line,
                                            do_start);
    }
};

template <>
struct log_stream_dispatch<false> {
    inline static null_stream exec(int lineloglevel, const char *file,
                                   const char *function, int line,
                                   bool do_start = true) {
        return null_stream();
    }
};

// if set OUTPUTLEVEL == LOG_NONE, disable logging
#if OUTPUTLEVEL == LOG_NONE
// totally disable logging
#define logger(lvl, fmt, ...)
#define logstream(lvl) \
    if (0) null_stream()
#else
#define logger(lvl, fmt, ...)                                                    \
    (log_dispatch<(lvl >= OUTPUTLEVEL)>::exec(lvl, __FILE__, __func__, __LINE__, \
                                              fmt, ##__VA_ARGS__))
#define logstream(lvl)                                                        \
    (log_stream_dispatch<(lvl >= OUTPUTLEVEL)>::exec(lvl, __FILE__, __func__, \
                                                     __LINE__))
#endif

// use LOG_endl just like std::endl
#define LOG_endl "\n"
