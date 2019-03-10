public class Results {

  public static ImmutableList<Supplier<Type>> fromStrings(Iterable<String> types) {
    return ImmutableList.copyOf(
        Iterables.transform(
            types,
            new Function<String, Supplier<Type>>() {
              @Override
              public Supplier<Type> apply(String input) {
                return Suppliers.typeFromString(input);
              }
            }));
  }

  public static com.google.errorprone.bugpatterns.apidiff.ApiDiffProto.EverythingDiff parseFrom(
      java.io.InputStream input) throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3.parseWithIOException(PARSER, input);
  }

  private static Level fromJdkLevel(java.util.logging.Level level) {
    if (level == java.util.logging.Level.ALL) {
      return Level.ALL;
    }
    if (level == java.util.logging.Level.FINER) {
      return Level.TRACE;
    }
    if (level == java.util.logging.Level.FINE) {
      return Level.DEBUG;
    }
    if (level == java.util.logging.Level.INFO) {
      return Level.INFO;
    }
    if (level == java.util.logging.Level.WARNING) {
      return Level.WARN;
    }
    if (level == java.util.logging.Level.SEVERE) {
      return Level.ERROR;
    }
    return Level.OFF;
  }

  public static com.google.errorprone.bugpatterns.apidiff.ApiDiffProto.MemberDiff parseFrom(
      java.io.InputStream input, com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3.parseWithIOException(
        PARSER, input, extensionRegistry);
  }

  public static com.google.errorprone.bugpatterns.apidiff.ApiDiffProto.Diff parseFrom(
      java.io.InputStream input, com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3.parseWithIOException(
        PARSER, input, extensionRegistry);
  }

  public static com.google.errorprone.bugpatterns.apidiff.ApiDiffProto.MemberDiff parseFrom(
      java.io.InputStream input) throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3.parseWithIOException(PARSER, input);
  }

  public static com.google.errorprone.bugpatterns.apidiff.ApiDiffProto.Diff parseFrom(
      java.io.InputStream input) throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3.parseWithIOException(PARSER, input);
  }

  public static com.google.errorprone.bugpatterns.apidiff.ApiDiffProto.Diff parseFrom(
      java.nio.ByteBuffer data) throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data);
  }

  public static com.google.errorprone.bugpatterns.apidiff.ApiDiffProto.MemberDiff parseFrom(
      java.nio.ByteBuffer data) throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data);
  }

  public static com.google.errorprone.bugpatterns.apidiff.ApiDiffProto.ClassDiff parseFrom(
      java.io.InputStream input) throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3.parseWithIOException(PARSER, input);
  }
}
