// connshim: redirect connect() to 127.0.0.1:<port> according to SNNI_CONNMAP="9530:19530,9531:19531".
// Only outgoing connections are rewritten; bind()/listen() are untouched, so each party still listens
// on its own port while its traffic to the peer passes through the shaping proxy.
#define _GNU_SOURCE
#include <dlfcn.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>

static int (*real_connect)(int, const struct sockaddr *, socklen_t);

static int mapped_port(int port) {
    const char *m = getenv("SNNI_CONNMAP");
    if (!m) return -1;
    char buf[256];
    strncpy(buf, m, sizeof(buf) - 1); buf[sizeof(buf) - 1] = 0;
    for (char *tok = strtok(buf, ","); tok; tok = strtok(NULL, ",")) {
        int from, to;
        char *c = strchr(tok, ':');
        if (!c) continue;
        *c = 0; from = atoi(tok); to = atoi(c + 1);
        if (from == port) return to;
    }
    return -1;
}

int connect(int fd, const struct sockaddr *addr, socklen_t len) {
    if (!real_connect) real_connect = dlsym(RTLD_NEXT, "connect");
    if (addr && addr->sa_family == AF_INET && len >= sizeof(struct sockaddr_in)) {
        const struct sockaddr_in *in = (const struct sockaddr_in *)addr;
        if (in->sin_addr.s_addr == htonl(INADDR_LOOPBACK)) {
            int to = mapped_port(ntohs(in->sin_port));
            if (to > 0) {
                struct sockaddr_in r = *in;
                r.sin_port = htons(to);
                return real_connect(fd, (const struct sockaddr *)&r, sizeof(r));
            }
        }
    }
    return real_connect(fd, addr, len);
}
