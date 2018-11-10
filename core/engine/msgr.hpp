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

#pragma once

#include <vector>

#include "query.hpp"

#include "comm/adaptor.hpp"

// utils
#include "logger2.hpp"

using namespace std;


class Messenger {
private:
    class Message {
    public:
        int sid;
        int tid;
        Bundle bundle;

        Message(int sid, int tid, Bundle &bundle)
            : sid(sid), tid(tid), bundle(bundle) { }
    };

    vector<Message> pending_msgs;

public:
    int sid;    // server id
    int tid;    // thread id

    Adaptor *adaptor;

    Messenger(int sid, int tid, Adaptor *adaptor) : sid(sid), tid(tid), adaptor(adaptor) { }

    inline void sweep_msgs() {
        if (!pending_msgs.size()) return;

        logstream(LOG_DEBUG) << "#" << tid << " "
                             << pending_msgs.size() << " pending msgs on engine." << LOG_endl;
        for (vector<Message>::iterator it = pending_msgs.begin(); it != pending_msgs.end();)
            if (adaptor->send(it->sid, it->tid, it->bundle))
                it = pending_msgs.erase(it);
            else
                ++it;
    }

    bool send_msg(Bundle &bundle, int dst_sid, int dst_tid) {
        if (adaptor->send(dst_sid, dst_tid, bundle))
            return true;

        // failed to send, then stash the msg to avoid deadlock
        pending_msgs.push_back(Message(dst_sid, dst_tid, bundle));
        return false;
    }

    Bundle recv_msg() { return adaptor->recv(); }

    bool tryrecv_msg(Bundle &bundle) { return adaptor->tryrecv(bundle); }

};