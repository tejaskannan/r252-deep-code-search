public class Results {

  private Object toFile(VisitorState state, Tree fileArg, SuggestedFix.Builder fix) {
    Type type = ASTHelpers.getType(fileArg);
    if (ASTHelpers.isSubtype(type, state.getSymtab().stringType, state)) {
      fix.addImport("java.io.file");
      return String.format("new file(%s)", state.getSourceForNode(fileArg));
    } else if (ASTHelpers.isSubtype(type, state.getTypeFromString("java.io.file"), state)) {
      return state.getSourceForNode(fileArg);
    } else {
      throw new AssertionError("unexpected type: " + type);
    }
  }

  public void writeFile(SourceFile file) {
    log.info(String.format("altered file %s thrown away", file.getPath()));
  }

  public void writeFile(SourceFile update) throws IOException {
    Path sourceFilePath = rootPath.resolve(update.getPath());
    String oldSource = new String(Files.readAllBytes(sourceFilePath), UTF_8);
    String newSource = update.getSourceText();
    if (!oldSource.equals(newSource)) {
      List<String> originalLines = LINE_SPLITTER.splitToList(oldSource);
      Patch<String> diff = DiffUtils.diff(originalLines, LINE_SPLITTER.splitToList(newSource));
      String relativePath = relativize(sourceFilePath);
      List<String> unifiedDiff =
          DiffUtils.generateUnifiedDiff(relativePath, relativePath, originalLines, diff, 2);
      String diffString = Joiner.on("\n").join(unifiedDiff) + "\n";
      diffByFile.put(sourceFilePath.toUri(), diffString);
    }
  }

  public void writeFile(SourceFile update) throws IOException {
    Path targetPath = rootPath.resolve(update.getPath());
    Files.write(targetPath, update.getSourceText().getBytes(StandardCharsets.UTF_8));
  }

  void loadProperties() {
    if (file != null && file.exists()) {
      try (InputStream in = new FileInputStream(file)) {
        properties.load(in);
        if (logger.isInfoEnabled()) {
          logger.info("load service store file " + file + ", data: " + properties);
        }
      } catch (Throwable e) {
        logger.warn("failed to load service store file " + file, e);
      }
    }
  }

  private static void writePatchFile(
      AtomicBoolean first, URI uri, PatchFileDestination fileDestination, Path patchFilePatch)
      throws IOException {
    String patchFile = fileDestination.patchFile(uri);
    if (patchFile != null) {
      if (first.compareAndSet(true, false)) {
        try {
          Files.deleteIfExists(patchFilePatch);
        } catch (IOException e) {
          throw new UncheckedIOException(e);
        }
      }
      Files.write(patchFilePatch, patchFile.getBytes(UTF_8), APPEND, CREATE);
    }
  }

  private void loadProperties() {
    if (file != null && file.exists()) {
      InputStream in = null;
      try {
        in = new FileInputStream(file);
        properties.load(in);
        if (logger.isInfoEnabled()) {
          logger.info("load registry cache file " + file + ", data: " + properties);
        }
      } catch (Throwable e) {
        logger.warn("failed to load registry cache file " + file, e);
      } finally {
        if (in != null) {
          try {
            in.close();
          } catch (IOException e) {
            logger.warn(e.getMessage(), e);
          }
        }
      }
    }
  }

  public JavaFileObject forResource(String fileName) {
    Preconditions.checkState(
        clazz.isPresent(), "clazz must be set if you want to add a source from a resource file");
    return forResource(clazz.get(), fileName);
  }

  public void writeTo(com.google.protobuf.CodedOutputStream output) throws java.io.IOException {
    if (!getClassNameBytes().isEmpty()) {
      com.google.protobuf.GeneratedMessageV3.writeString(output, 1, className_);
    }
    unknownFields.writeTo(output);
  }

  public void writeTo(com.google.protobuf.CodedOutputStream output) throws java.io.IOException {
    for (int i = 0; i < classDiff_.size(); i++) {
      output.writeMessage(1, classDiff_.get(i));
    }
    unknownFields.writeTo(output);
  }
}
