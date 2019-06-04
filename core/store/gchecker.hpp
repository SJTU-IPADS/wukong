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

// store
#include "gstore.hpp"

class GChecker {
    GStore *gstore;

    volatile uint64_t ivertex_num = 0;
    volatile uint64_t nvertex_num = 0;

    void check2_idx_in(ikey_t key) {
        uint64_t vsz = 0;
        // get all local types
        edge_t *vres = gstore->get_edges_local(0, 0, TYPE_ID, OUT, vsz);
        bool found = false;
        // check whether the pid exists or duplicate
        for (int i = 0; i < vsz; i++) {
            if (vres[i].val == key.pid && !found) {
                found = true;
            } else if (vres[i].val == key.pid && found) {
                logstream(LOG_ERROR) << "In the value part of all local types [ 0 | TYPE_ID | OUT ]"
                                     << " there is duplicate value " << key.pid << LOG_endl;
            }
        }

        // pid does not exist in local types, maybe it is predicate
        if (!found) {
            uint64_t psz = 0;
            // get all local predicates
            edge_t *pres = gstore->get_edges_local(0, 0, PREDICATE_ID, OUT, psz);
            bool found = false;
            // check whether the pid exists or duplicate
            for (int i = 0; i < psz; i++) {
                if (pres[i].val == key.pid && !found) {
                    found = true;
                } else if (pres[i].val == key.pid && found) {
                    logstream(LOG_ERROR) << "In the value part of all local predicates [ 0 | PREDICATE_ID | OUT ]"
                                         << " there is duplicate value " << key.pid << LOG_endl;
                    break;
                }
            }

            if (!found) {
                logstream(LOG_ERROR) << "if " << key.pid << "is predicate, "
                                     << "in the value part of all local predicates [ 0 | PREDICATE_ID | OUT ]"
                                     << " there is NO value " << key.pid << LOG_endl;
                logstream(LOG_ERROR) << "if " << key.pid << " is type, "
                                     << "in the value part of all local types [ 0 | TYPE_ID | OUT ]"
                                     << " there is NO value " << key.pid << LOG_endl;
            }

            uint64_t vsz = 0;
            // get the vid refered which refered by the type/predicate
            edge_t *vres = gstore->get_edges_local(0, 0, key.pid, IN, vsz);
            if (vsz == 0) {
                logstream(LOG_ERROR) << "if " << key.pid << " is type, "
                                     << "in the value part of all local types [ 0 | TYPE_ID | OUT ]"
                                     << " there is NO value " << key.pid << LOG_endl;
                return;
            }

            for (int i = 0; i < vsz; i++) {
                found = false;
                uint64_t sosz = 0;
                // get all local objects/subjects
                edge_t *sores = gstore->get_edges_local(0, 0, TYPE_ID, IN, sosz);
                for (int j = 0; j < sosz; j++) {
                    if (sores[j].val == vres[i].val && !found) {
                        found = true;
                    } else if (sores[j].val == vres[i].val && found) {
                        logstream(LOG_ERROR) << "In the value part of all local subjects/objects [ 0 | TYPE_ID | IN ]"
                                             << " there is duplicate value " << vres[i].val << LOG_endl;
                        break;
                    }
                }

                if (!found)
                    logstream(LOG_ERROR) << "In the value part of all local subjects/objects [ 0 | TYPE_ID | IN ]"
                                         << " there is no value " << vres[i].val << LOG_endl;

                found = false;
                uint64_t p2sz = 0;
                // get vid's all predicate
                edge_t *p2res = gstore->get_edges_local(0, vres[i].val, PREDICATE_ID, OUT, p2sz);
                for (int j = 0; j < p2sz; j++) {
                    if (p2res[j].val == key.pid && !found) {
                        found = true;
                    } else if (p2res[j].val == key.pid && found) {
                        logstream(LOG_ERROR) << "In the value part of " << vres[i].val << "'s all predicates [ "
                                             << vres[i].val << " | PREDICATE_ID | OUT ], there is duplicate value "
                                             << key.pid << LOG_endl;
                        break;
                    }
                }

                if (!found)
                    logstream(LOG_ERROR) << "In the value part of " << vres[i].val << "'s all predicates [ "
                                         << vres[i].val << " | PREDICATE_ID | OUT ], there is no value "
                                         << key.pid << LOG_endl;
            }
        }
    }

    // check (in) predicate/type index vertices (1 and 2)
    void check_idx_in(ikey_t key) {
        uint64_t vsz = 0;
        // get the vids which refered by index
        edge_t *vres = gstore->get_edges_local(0, key.vid, key.pid, (dir_t)key.dir, vsz);
        for (int i = 0; i < vsz; i++) {
            uint64_t tsz = 0;
            // get the vids's type
            edge_t *tres = gstore->get_edges_local(0, vres[i].val, TYPE_ID, OUT, tsz);
            bool found = false;
            for (int j = 0; j < tsz; j++) {
                if (tres[j].val == key.pid && !found)
                    found = true;
                else if (tres[j].val == key.pid && found)  // duplicate type
                    logstream(LOG_ERROR) << "In the value part of normal key/value pair "
                                         << "[ " << key.vid << " | TYPE_ID | OUT], "
                                         << "there is DUPLICATE type " << key.pid << LOG_endl;
            }

            // may be it is a predicate_index
            if (tsz != 0 && !found) {
                // check if the key generated by vid and pid exists
                if (gstore->get_vertex_local(0, ikey_t(vres[i].val, key.pid, OUT)).key.is_empty()) {
                    logstream(LOG_ERROR) << "if " << key.pid << " is type id, then there is NO type "
                                         << key.pid << " in normal key/value pair ["
                                         << key.vid << " | TYPE_ID | OUT]'s value part" << LOG_endl;
                    logstream(LOG_ERROR) << "And if " << key.pid << " is predicate id, "
                                         << " then there is NO key called "
                                         << "[ " << vres[i].val << " | " << key.pid << " | " << "] exist."
                                         << LOG_endl;
                }
            }
        }
        wukong::atomic::add_and_fetch(&ivertex_num, 1);

#ifdef VERSATILE
        check2_idx_in(key);
#endif
    }

    void check2_idx_out(ikey_t key) {
        uint64_t psz = 0;
        // get all local predicates
        edge_t *pres = gstore->get_edges_local(0, 0, PREDICATE_ID, OUT, psz);
        bool found = false;
        // check whether the pid exists or duplicate
        for (int i = 0; i < psz; i++) {
            if (pres[i].val == key.pid && !found) {
                found = true;
            } else if (pres[i].val == key.pid && found) {
                logstream(LOG_ERROR) << "In the value part of all local predicates [ 0 | PREDICATE_ID | OUT ]"
                                     << " there is duplicate value " << key.pid << LOG_endl;
                break;
            }
        }

        if (!found)
            logstream(LOG_ERROR) << "In the value part of all local predicates [ 0 | PREDICATE_ID | OUT ]"
                                 << " there is no value " << key.pid << LOG_endl;

        uint64_t vsz = 0;
        // get the vid refered which refered by the predicate
        edge_t *vres = gstore->get_edges_local(0, 0, key.pid, OUT, vsz);
        for (int i = 0; i < vsz; i++) {
            found = false;
            uint64_t sosz = 0;
            // get all local objects/subjects
            edge_t *sores = gstore->get_edges_local(0, 0, TYPE_ID, IN, sosz);
            for (int j = 0; j < sosz; j++) {
                if (sores[j].val == vres[i].val && !found) {
                    found = true;
                } else if (sores[j].val == vres[i].val && found) {
                    logstream(LOG_ERROR) << "In the value part of all local subjects/objects [ 0 | TYPE_ID | IN ]"
                                         << " there is duplicate value " << vres[i].val << LOG_endl;
                    break;
                }
            }

            if (!found)
                logstream(LOG_ERROR) << "In the value part of all local subjects/objects [ 0 | TYPE_ID | IN ]"
                                     << " there is no value " << vres[i].val << LOG_endl;

            found = false;
            uint64_t psz = 0;
            // get vid's all predicate
            edge_t *pres = gstore->get_edges_local(0, vres[i].val, PREDICATE_ID, IN, psz);
            for (int j = 0; j < psz; j++) {
                if (pres[j].val == key.pid && !found) {
                    found = true;
                } else if (pres[j].val == key.pid && found) {
                    logstream(LOG_ERROR) << "In the value part of " << vres[i].val << "'s all predicates [ "
                                         << vres[i].val << "PREDICATE_ID | IN ], there is duplicate value "
                                         << key.pid << LOG_endl;
                    break;
                }
            }

            if (!found)
                logstream(LOG_ERROR) << "In the value part of " << vres[i].val << "'s all predicates [ "
                                     << vres[i].val << "PREDICATE_ID | IN ], there is no value "
                                     << key.pid << LOG_endl;
        }
    }

    // check (out) predicate index vertices (1)
    void check_idx_out(ikey_t key) {
        uint64_t vsz = 0;
        // get the vids which refered by predicate index
        edge_t *vres = gstore->get_edges_local(0, key.vid, key.pid, (dir_t)key.dir, vsz);
        for (int i = 0; i < vsz; i++)
            // check if the key generated by vid and pid exists
            if (gstore->get_vertex_local(0, ikey_t(vres[i].val, key.pid, IN)).key.is_empty())
                logstream(LOG_ERROR) << "The key " << " [ " << vres[i].val << " | "
                                     << key.pid << " | " << " IN ] does not exist." << LOG_endl;
        wukong::atomic::add_and_fetch(&ivertex_num, 1);

#ifdef VERSATILE
        check2_idx_out(key);
#endif
    }

    void check2_type(ikey_t key) {
        bool found = false;
        uint64_t psz = 0;
        // get vid' all predicates
        edge_t *pres = gstore->get_edges_local(0, key.vid, PREDICATE_ID, OUT, psz);
        // check if there is TYPE_ID in vid's predicates
        for (int i = 0; i < psz; i++) {
            if (pres[i].val == key.pid && !found)
                found = true;
            else if (pres[i].val == key.pid && found) {
                logstream(LOG_ERROR) << "In the value part of "
                                     << key.vid << "'s all predicates [ "
                                     << key.vid << "PREDICATE_ID | OUT ], there is DUPLICATE value "
                                     << key.pid << LOG_endl;
                break;
            }
        }

        if (!found)
            logstream(LOG_ERROR) << "In the value part of "
                                 << key.vid << "'s all predicates [ "
                                 << key.vid << "PREDICATE_ID | OUT ], there is NO value "
                                 << key.pid << LOG_endl;

        found = false;
        uint64_t ossz = 0;
        // get all local subjects/objects
        edge_t *osres = gstore->get_edges_local(0, 0, key.pid, IN, ossz);
        for (int i = 0; i < ossz; i++) {
            if (osres[i].val == key.vid && !found)
                found = true;
            else if (osres[i].val == key.vid && found) {
                logstream(LOG_ERROR) << "In the value part of all local subjects/objects [ 0 | TYPE_ID | IN ]"
                                     << " there is DUPLICATE value " << key.vid << LOG_endl;
                break;
            }
        }

        if (!found)
            logstream(LOG_ERROR) << "In the value part of all local subjects/objects [ 0 | TYPE_ID | IN ]"
                                 << " there is NO value " << key.vid << LOG_endl;
    }

    // check normal types (7)
    void check_type(ikey_t key) {
        uint64_t tsz = 0;
        // get the vid's all type
        edge_t *tres = gstore->get_edges_local(0, key.vid, key.pid, (dir_t)key.dir, tsz);
        for (int i = 0; i < tsz; i++) {
            uint64_t vsz = 0;
            // get the vids which refered by the type
            edge_t *vres = gstore->get_edges_local(0, 0, tres[i].val, IN, vsz);
            bool found = false;
            for (int j = 0; j < vsz; j++) {
                if (vres[j].val == key.vid && !found) {
                    found = true;
                } else if (vres[j].val == key.vid && found) { // duplicate vid
                    logstream(LOG_ERROR) << "In the value part of type index "
                                         << "[ 0 | " << tres[i].val << " | IN ], "
                                         << "there is duplicate value " << key.vid << LOG_endl;
                }
            }

            if (!found) // vid miss
                logstream(LOG_ERROR) << "In the value part of type index "
                                     << "[ 0 | " << tres[i].val << " | IN ], "
                                     << "there is no value " << key.vid << LOG_endl;
        }
        wukong::atomic::add_and_fetch(&nvertex_num, 1);

#ifdef VERSATILE
        check2_type(key);
#endif
    }

    // check normal vertices (6)
    void check_normal(ikey_t key, dir_t dir) {
        uint64_t vsz = 0;
        // get the vids which refered by the predicated
        edge_t *vres = gstore->get_edges_local(0, 0, key.pid, dir, vsz);
        bool found = false;
        for (int i = 0; i < vsz; i++) {
            if (vres[i].val == key.vid && !found) {
                found = true;
            } else if (vres[i].val == key.vid && found) { //duplicate vid
                logstream(LOG_ERROR) << "In the value part of predicate index "
                                     << "[ 0 | " << key.pid << " | " << dir << " ], "
                                     << "there is duplicate value " << key.vid << LOG_endl;
                break;
            }
        }

        if (!found) // vid miss
            logstream(LOG_ERROR) << "In the value part of predicate index "
                                 << "[ 0 | " << key.pid << " | " << dir << " ], "
                                 << "there is no value " << key.vid << LOG_endl;
        wukong::atomic::add_and_fetch(&nvertex_num, 1);
    }

    void check(ikey_t key, bool index, bool normal) {
        if (key.vid == 0 && is_tpid(key.pid) && key.dir == IN) {               // (2) and (1)
            if (index) check_idx_in(key);
        } else if (key.vid == 0 && is_tpid(key.pid) && key.dir == OUT) {       // (1)
            if (index) check_idx_out(key);
        } else if (is_vid(key.vid) && key.pid == TYPE_ID && key.dir == OUT) {  // (7)
            if (normal) check_type(key);
        } else if (is_vid(key.vid) && is_tpid(key.pid) && key.dir == OUT) {    // (6)
            if (normal) check_normal(key, IN);
        } else if (is_vid(key.vid) && is_tpid(key.pid) && key.dir == IN) {     // (6)
            if (normal) check_normal(key, OUT);
        }
    }

public:
    GChecker(GStore *gstore): gstore(gstore) {}

    int gstore_check(bool index, bool normal) {
        logstream(LOG_INFO) << "Graph storage intergity check has started on server "
                            << gstore->sid << LOG_endl;
        ivertex_num = 0;
        nvertex_num = 0;
        uint64_t total_buckets = gstore->num_buckets + gstore->num_buckets_ext;
        uint64_t cnt_flag = (total_buckets / 20) + 1;
        uint64_t buckets_count = 0;

        #pragma omp parallel for num_threads(Global::num_engines)
        for (uint64_t bucket_id = 0; bucket_id < total_buckets; bucket_id++) {
            uint64_t slot_id = bucket_id * GStore::ASSOCIATIVITY;
            for (int i = 0; i < GStore::ASSOCIATIVITY - 1; i++, slot_id++)
                if (!gstore->vertices[slot_id].key.is_empty())
                    check(gstore->vertices[slot_id].key, index, normal);

            uint64_t current_count = wukong::atomic::add_and_fetch(&buckets_count, 1);
            if (current_count % cnt_flag == 0) {
                logstream(LOG_INFO) << "Server#" << gstore->sid << " already check "
                                    << (current_count / cnt_flag) * 5 << "%" << LOG_endl;
            }
        }

        logstream(LOG_INFO) << "Server#" << gstore->sid << " has checked "
                            << ivertex_num << " index vertices and "
                            << nvertex_num << " normal vertices." << LOG_endl;
        return 0;
    }
};
