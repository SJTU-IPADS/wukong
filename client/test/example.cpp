#include <algorithm>
#include <fstream>
#include <sstream>
#include "console.h"
using namespace std;

Console *Console::instance = NULL;
char **Console::tab_commands = NULL;

int main(int argc, char *argv[]) {
	if (argc != 2) {
		cout << "usage: ./wukong server_file hostfile" << endl;
		exit(-1);
	}
	Console::getInstance()->setServer(argv[1]);
	Console::getInstance()->run();
	Console::deleteInstance();

	// char address[30];
	// zmq::context_t context1(1);
	// zmq::context_t context2(1);
	// zmq::socket_t receiver(context1, ZMQ_REP);
	// s_set_id(receiver);
	// sprintf(address, "tcp://*:%d", 8123);
	// receiver.bind(address);

	// zmq::socket_t sender(context2, ZMQ_REQ);
	// sprintf(address, "tcp://127.0.0.1:8123");
	// sender.connect(address);

	// ifstream ist("./libwukongapi.so", ios::binary);
	// std::string str((std::istreambuf_iterator<char>(ist)),  std::istreambuf_iterator<char>());


	// s_send(sender, str);
	// string request = s_recv(receiver);
	// ofstream ost("./libaaa.so", ios::binary);
	// ost<<request;

	// s_send(receiver,"hi");
	// request = s_recv(sender);
	// cout<<request<<endl;
	return 0;
}