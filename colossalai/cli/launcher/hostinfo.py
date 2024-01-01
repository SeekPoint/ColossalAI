import socket
from pydebug import gd, infoTensor

class HostInfo:
    """
    A data class to store host connection-related data.

    Args:
        hostname (str): name or IP address of the host
        port (str): the port for ssh connection
    """

    def __init__(
        self,
        hostname: str,
        port: str = None,
    ):
        gd.debuginfo(prj="mt", info=f'__FUNC_IN_OUT__')
        self.hostname = hostname
        self.port = port
        self.is_local_host = HostInfo.is_host_localhost(hostname, port)
        # 一台机器两张卡，一个true，一个false
        gd.debuginfo(prj="mt", info=f'self.is_local_host={self.is_local_host}')
        gd.debuginfo(prj="mt", info=f'__FUNC_IN_OUT__')

    @staticmethod
    def is_host_localhost(hostname: str, port: str = None) -> None:
        """
        Check if the host refers to the local machine.

        Args:
            hostname (str): name or IP address of the host
            port (str): the port for ssh connection

        Returns:
            bool: True if it is local, False otherwise
        """
        gd.debuginfo(prj="mt", info=f'')
        if port is None:
            gd.debuginfo(prj="mt", info=f'')
            port = 22  # no port specified, lets just use the ssh port

        # socket.getfqdn("127.0.0.1") does not return localhost
        # on some users' machines
        # thus, we directly return True if hostname is localhost, 127.0.0.1 or 0.0.0.0
        if hostname in ("localhost", "127.0.0.1", "0.0.0.0"):
            gd.debuginfo(prj="mt", info=f'')
            return True

        hostname = socket.getfqdn(hostname)
        localhost = socket.gethostname()
        localaddrs = socket.getaddrinfo(localhost, port)
        targetaddrs = socket.getaddrinfo(hostname, port)

        '''
        
mt F# /ColossalAI/colossalai/cli/launcher/hostinfo.py L#: 54 f# is_host_localhost I# hostname=MZ32-00
mt F# /ColossalAI/colossalai/cli/launcher/hostinfo.py L#: 55 f# is_host_localhost I# localhost=MZ32-00
hostinfo.py L#: 56 f# is_host_localhost I# localaddrs=[(<AddressFamily.AF_INET: 2>, <SocketKind.SOCK_STREAM: 1>, 6, '', ('192.168.1.11', 22)), (<AddressFamily.AF_INET: 2>, <SocketKind.SOCK_DGRAM: 2>, 17, '', ('192.168.1.11', 22)), (<AddressFamily.AF_INET: 2>, <SocketKind.SOCK_RAW: 3>, 0, '', ('192.168.1.11', 22))]
hostinfo.py L#: 57 f# is_host_localhost I#targetaddrs=[(<AddressFamily.AF_INET: 2>, <SocketKind.SOCK_STREAM: 1>, 6, '', ('192.168.1.11', 22)), (<AddressFamily.AF_INET: 2>, <SocketKind.SOCK_DGRAM: 2>, 17, '', ('192.168.1.11', 22)), (<AddressFamily.AF_INET: 2>, <SocketKind.SOCK_RAW: 3>, 0, '', ('192.168.1.11', 22))]

        '''

        gd.debuginfo(prj="mt", info=f'hostname={hostname}')
        gd.debuginfo(prj="mt", info=f'localhost={localhost}')
        gd.debuginfo(prj="mt", info=f'localaddrs={localaddrs}')
        gd.debuginfo(prj="mt", info=f'targetaddrs={targetaddrs}')

        return localaddrs == targetaddrs

    def __str__(self):
        return f"hostname: {self.hostname}, port: {self.port}"

    def __repr__(self):
        return self.__str__()


class HostInfoList:
    """
    A data class to store a list of HostInfo objects.
    """

    def __init__(self):
        gd.debuginfo(prj="mt", info=f'__FUNC_IN_OUT__')
        self.hostinfo_list = []
        gd.debuginfo(prj="mt", info=f'__FUNC_IN_OUT__')

    def append(self, hostinfo: HostInfo) -> None:
        """
        Add an HostInfo object to the list.

        Args:
            hostinfo (HostInfo): host information
        """
        gd.debuginfo(prj="mt", info=f'')
        self.hostinfo_list.append(hostinfo)

    def remove(self, hostname: str) -> None:
        """
        Add an HostInfo object to the list.

        Args:
            hostname (str): the name of the host
        """
        gd.debuginfo(prj="mt", info=f'')
        hostinfo = self.get_hostinfo(hostname)
        self.hostinfo_list.remove(hostinfo)

    def get_hostinfo(self, hostname: str) -> HostInfo:
        """
        Return the HostInfo object which matches with the hostname.

        Args:
            hostname (str): the name of the host

        Returns:
            hostinfo (HostInfo): the HostInfo object which matches with the hostname
        """
        gd.debuginfo(prj="mt", info=f'')
        for hostinfo in self.hostinfo_list:
            if hostinfo.hostname == hostname:
                return hostinfo

        raise Exception(f"Hostname {hostname} is not found")

    def has(self, hostname: str) -> bool:
        """
        Check if the hostname has been added.

        Args:
            hostname (str): the name of the host

        Returns:
            bool: True if added, False otherwise
        """
        gd.debuginfo(prj="mt", info=f'')
        for hostinfo in self.hostinfo_list:
            if hostinfo.hostname == hostname:
                return True
        return False

    def __iter__(self):
        return iter(self.hostinfo_list)

    def __len__(self):
        return len(self.hostinfo_list)
