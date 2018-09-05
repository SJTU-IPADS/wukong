# GPU related unit test

# 1 test-adaptor

In this test, a server will send data on gpu to another server's ring buffer with adaptor's `send_dev2host` interface.

If everything is fine, the output will be like this:

```
ERROR:    ---I am sender, on GPU: 1, 4, 3
ERROR:    ---type: 3, 1 4 3
```

## How to use

- modify the `mpd.hosts`
- modify `CMakeLists.txt`, make sure the source contains `test-adaptor.cpp`
- `./build.sh`
- `./sync.sh`
- `./run.sh 2`
