workspace(name = "aibench")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

# proto_library rules implicitly depend on @com_google_protobuf//:protoc,
# which is the proto-compiler.
# This statement defines the @com_google_protobuf repo.
http_archive(
    name = "com_google_protobuf",
    sha256 = "283cd18f82341e96afd75fa8b85e47680b47b7db1399a1a944a1b53bfb48b27f",
    strip_prefix = "protobuf-3.20.3",
    urls = [
        "https://github.com/protocolbuffers/protobuf/releases/download/v3.20.3/protobuf-all-3.20.3.zip",
    ],
)

load("@com_google_protobuf//:protobuf_deps.bzl", "PROTOBUF_MAVEN_ARTIFACTS", "protobuf_deps")
protobuf_deps()

http_archive(
    name = "gtest",
    build_file = "@aibench//:third_party/googletest/googletest.BUILD",
    sha256 = "f3ed3b58511efd272eb074a3a6d6fb79d7c2e6a0e374323d1e6bcbcc1ef141bf",
    strip_prefix = "googletest-release-1.8.0",
    urls = [
        "https://cnbj1.fds.api.xiaomi.com/mace/third-party/googletest/googletest-release-1.8.0.zip",
        "https://github.com/google/googletest/archive/release-1.8.0.zip",
    ],
)

http_archive(
    name = "opencv",
    build_file = "@aibench//:third_party/opencv/opencv.BUILD",
    sha256 = "9e4350a7aa5f4c8600a1a94466d42546098d76378f24bd2cf05ab7e96959c910",
    urls = [
        "http://cnbj1.fds.api.xiaomi.com/aibench/third_party/opencv-ndk-custom-3.4.4.zip",
    ],
)


http_archive(
    name = "ncnn",
    build_file = "@aibench//:third_party/ncnn/ncnn.BUILD",
    sha256 = "31b04206847d18547c4d5bc8874fc4fe89b61ca0fad7e2e00a486d9aca0639e6",
    strip_prefix = "ncnn-20220729",
    type = "zip",
    urls = [
        "https://github.com/Tencent/ncnn/archive/refs/tags/20220729.zip",
    ],
)

# You need to comment following http_archive and uncomment following
# new_local_repository to benchmark SNPE
http_archive(
    name = "snpe",
    build_file = "@aibench//:third_party/snpe/snpe.BUILD",
    sha256 = "88ccf182b47c8ee36e338fd3d6f207661e95992f5bbdacc2867320b1eb817034",
    strip_prefix = "snpe-1.50.0.2622",
    urls = [
        "http://cnbj1-fds.api.xiaomi.net/aibench/third_party/snpe-1.50.0.zip",
    ],
)
# You need to uncomment following new_local_repository and comment foregoing
# http_archive to benchmark SNPE
# new_local_repository(
#     name = "snpe",
#     build_file = "@aibench//:third_party/snpe/snpe.BUILD",
#     path = "/path/to/snpe",
# )

http_archive(
    # v2.2.0 + fix of include path
    name = "com_github_gflags_gflags",
    sha256 = "16903f6bb63c00689eee3bf7fb4b8f242934f6c839ce3afc5690f71b712187f9",
    strip_prefix = "gflags-30dbc81fb5ffdc98ea9b14b1918bfe4e8779b26e",
    urls = [
        "https://cnbj1.fds.api.xiaomi.com/mace/third-party/gflags/gflags-30dbc81fb5ffdc98ea9b14b1918bfe4e8779b26e.zip",
        "https://github.com/gflags/gflags/archive/30dbc81fb5ffdc98ea9b14b1918bfe4e8779b26e.zip",
    ],
)

http_archive(
    name = "six_archive",
    build_file = "@aibench//:third_party/six/six.BUILD",
    sha256 = "105f8d68616f8248e24bf0e9372ef04d3cc10104f1980f54d57b2ce73a5ad56a",
    strip_prefix = "six-1.10.0",
    urls = [
        "https://cnbj1.fds.api.xiaomi.com/mace/third-party/six/six-1.10.0.tar.gz",
        "http://mirror.bazel.build/pypi.python.org/packages/source/s/six/six-1.10.0.tar.gz",
        "https://pypi.python.org/packages/source/s/six/six-1.10.0.tar.gz",
    ],
)

bind(
    name = "six",
    actual = "@six_archive//:six",
)

bind(
    name = "gflags",
    actual = "@com_github_gflags_gflags//:gflags",
)

bind(
    name = "gflags_nothreads",
    actual = "@com_github_gflags_gflags//:gflags_nothreads",
)

# Set up Android NDK
android_ndk_repository(
    name = "androidndk",
    # Android 7.0, without HIAI the api_level can be 21
    api_level = 24,
)

# Set up default cross compilers for arm linux
http_archive(
    name = "gcc_linaro_7_3_1_arm_linux_gnueabihf",
    build_file = "@aibench//:third_party/compilers/arm_compiler.BUILD",
    sha256 = "7248bf105d0d468887a9b8a7120bb281ac8ad0223d9cb3d00dc7c2d498485d91",
    strip_prefix = "gcc-linaro-7.3.1-2018.05-x86_64_arm-linux-gnueabihf",
    urls = [
        "https://cnbj1.fds.api.xiaomi.com/mace/third-party/gcc-linaro/gcc-linaro-7.3.1-2018.05-x86_64_arm-linux-gnueabihf.tar.xz",
        "https://releases.linaro.org/components/toolchain/binaries/latest/arm-linux-gnueabihf/gcc-linaro-7.3.1-2018.05-x86_64_arm-linux-gnueabihf.tar.xz",
    ],
)

http_archive(
    name = "gcc_linaro_7_3_1_aarch64_linux_gnu",
    build_file = "@aibench//:third_party/compilers/aarch64_compiler.BUILD",
    sha256 = "73eed74e593e2267504efbcf3678918bb22409ab7afa3dc7c135d2c6790c2345",
    strip_prefix = "gcc-linaro-7.3.1-2018.05-x86_64_aarch64-linux-gnu",
    urls = [
        "https://cnbj1.fds.api.xiaomi.com/mace/third-party/gcc-linaro/gcc-linaro-7.3.1-2018.05-x86_64_aarch64-linux-gnu.tar.xz",
        "https://releases.linaro.org/components/toolchain/binaries/latest/aarch64-linux-gnu/gcc-linaro-7.3.1-2018.05-x86_64_aarch64-linux-gnu.tar.xz",
    ],
)

# You need to comment following http_archive and uncomment following
# new_local_repository to benchmark HIAI
http_archive(
    name = "hiai",
    build_file = "@aibench//:third_party/hiai/hiai.BUILD",
    sha256 = "8da8305617573bc495df8f4509fcb1655ffb073d790d9c0b6ca32ba4a4e41055",
    strip_prefix = "HiAI_DDK_100.200.010.011",
    type = "zip",
    urls = [
        "http://cnbj1-fds.api.xiaomi.net/aibench/third_party/HiAI_DDK_100.200.010.011_LITE.zip",
    ],
)

# new_local_repository to benchmark TNN
http_archive(
    name = "tnn",
    build_file = "@aibench//:third_party/tnn/tnn.BUILD",
    sha256 = "83d084ca0a8974987f0f57c9b78e128d70499e00b0893f371b055fa75643c6db",
    strip_prefix = "tnn-v0.3.0",
    urls = [
        "http://cnbj1-fds.api.xiaomi.net/aibench/third_party/tnn-v0.3.0.zip",
    ],
)
# You need to uncomment following new_local_repository and comment foregoing
# http_archive to benchmark HIAI
# new_local_repository(
#     name = "hiai",
#     build_file = "@aibench//:third_party/hiai/hiai.BUILD",
#     path = "/path/to/hiai",
# )
