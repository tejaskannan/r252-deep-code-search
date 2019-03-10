public class Results {

  public void describeTo(Description description) {
    description.appendText("a diagnostic on line ").appendValue(line);
  }

  public static Matcher<Diagnostic<? extends JavaFileObject>> diagnosticOnLine(
      final URI fileURI, final long line) {
    return new TypeSafeDiagnosingMatcher<Diagnostic<? extends JavaFileObject>>() {
      @Override
      public boolean matchesSafely(
          Diagnostic<? extends JavaFileObject> item, Description mismatchDescription) {
        if (item.getSource() == null) {
          mismatchDescription
              .appendText("diagnostic not attached to a file: ")
              .appendValue(item.getMessage(ENGLISH));
          return false;
        }
        if (!item.getSource().toUri().equals(fileURI)) {
          mismatchDescription.appendText("diagnostic not in file ").appendValue(fileURI);
          return false;
        }
        if (item.getLineNumber() != line) {
          mismatchDescription
              .appendText("diagnostic not on line ")
              .appendValue(item.getLineNumber());
          return false;
        }
        return true;
      }

      @Override
      public void describeTo(Description description) {
        description.appendText("a diagnostic on line ").appendValue(line);
      }
    };
  }

  public void describeTo(Description description) {
    description
        .appendText("a diagnostic on line ")
        .appendValue(line)
        .appendText(" that matches \n")
        .appendValue(matcher)
        .appendText("\n");
  }

  public static Matcher<Diagnostic<? extends JavaFileObject>> diagnosticOnLine(
      final URI fileURI, final long line, final Predicate<? super String> matcher) {
    return new TypeSafeDiagnosingMatcher<Diagnostic<? extends JavaFileObject>>() {
      @Override
      public boolean matchesSafely(
          Diagnostic<? extends JavaFileObject> item, Description mismatchDescription) {
        if (item.getSource() == null) {
          mismatchDescription
              .appendText("diagnostic not attached to a file: ")
              .appendValue(item.getMessage(ENGLISH));
          return false;
        }
        if (!item.getSource().toUri().equals(fileURI)) {
          mismatchDescription.appendText("diagnostic not in file ").appendValue(fileURI);
          return false;
        }
        if (item.getLineNumber() != line) {
          mismatchDescription
              .appendText("diagnostic not on line ")
              .appendValue(item.getLineNumber());
          return false;
        }
        if (!matcher.apply(item.getMessage(Locale.getDefault()))) {
          mismatchDescription.appendText("diagnostic does not match ").appendValue(matcher);
          return false;
        }
        return true;
      }

      @Override
      public void describeTo(Description description) {
        description
            .appendText("a diagnostic on line ")
            .appendValue(line)
            .appendText(" that matches \n")
            .appendValue(matcher)
            .appendText("\n");
      }
    };
  }

  public void describeTo(Description description) {
    description
        .appendText("a diagnostic on line:column ")
        .appendValue(line)
        .appendText(":")
        .appendValue(column);
  }

  public boolean matchesSafely(
      Diagnostic<? extends JavaFileObject> item, Description mismatchDescription) {
    if (item.getSource() == null) {
      mismatchDescription
          .appendText("diagnostic not attached to a file: ")
          .appendValue(item.getMessage(ENGLISH));
      return false;
    }
    if (!item.getSource().toUri().equals(fileURI)) {
      mismatchDescription.appendText("diagnostic not in file ").appendValue(fileURI);
      return false;
    }
    if (item.getLineNumber() != line) {
      mismatchDescription.appendText("diagnostic not on line ").appendValue(item.getLineNumber());
      return false;
    }
    return true;
  }

  public SourceFile readFile(String path) throws IOException {
    return new SourceFile(
        path, new String(Files.readAllBytes(rootPath.resolve(path)), StandardCharsets.UTF_8));
  }

  public static Matcher<Diagnostic<? extends JavaFileObject>> diagnosticLineAndColumn(
      final long line, final long column) {
    return new TypeSafeDiagnosingMatcher<Diagnostic<? extends JavaFileObject>>() {
      @Override
      protected boolean matchesSafely(
          Diagnostic<? extends JavaFileObject> item, Description mismatchDescription) {
        if (item.getLineNumber() != line) {
          mismatchDescription
              .appendText("diagnostic not on line ")
              .appendValue(item.getLineNumber());
          return false;
        }
        if (item.getColumnNumber() != column) {
          mismatchDescription
              .appendText("diagnostic not on column ")
              .appendValue(item.getColumnNumber());
          return false;
        }
        return true;
      }

      @Override
      public void describeTo(Description description) {
        description
            .appendText("a diagnostic on line:column ")
            .appendValue(line)
            .appendText(":")
            .appendValue(column);
      }
    };
  }

  private void changed() throws RemotingException {
    try {
      String[] lines = IOUtils.readLines(file);
      for (String line : lines) {
        connect(URL.valueOf(line));
      }
    } catch (IOException e) {
      throw new RemotingException(
          new InetSocketAddress(NetUtils.getLocalHost(), 0),
          getUrl().toInetSocketAddress(),
          e.getMessage(),
          e);
    }
  }

  public boolean matchesSafely(
      Diagnostic<? extends JavaFileObject> item, Description mismatchDescription) {
    if (item.getSource() == null) {
      mismatchDescription
          .appendText("diagnostic not attached to a file: ")
          .appendValue(item.getMessage(ENGLISH));
      return false;
    }
    if (!item.getSource().toUri().equals(fileURI)) {
      mismatchDescription.appendText("diagnostic not in file ").appendValue(fileURI);
      return false;
    }
    if (item.getLineNumber() != line) {
      mismatchDescription.appendText("diagnostic not on line ").appendValue(item.getLineNumber());
      return false;
    }
    if (!matcher.apply(item.getMessage(Locale.getDefault()))) {
      mismatchDescription.appendText("diagnostic does not match ").appendValue(matcher);
      return false;
    }
    return true;
  }
}
