public class Results {

  public java.net.URL toJavaURL() {
    return super.toJavaURL();
  }

  static MatchedComment notAnnotated() {
    return new AutoValue_NamedParameterComment_MatchedComment(
        new Comment() {
          @Override
          public String getText() {
            throw new IllegalArgumentException(
                "attempt to call gettext on comment when in not_annotated state");
          }

          @Override
          public int getSourcePos(int i) {
            throw new IllegalArgumentException(
                "attempt to call gettext on comment when in not_annotated state");
          }

          @Override
          public CommentStyle getStyle() {
            throw new IllegalArgumentException(
                "attempt to call gettext on comment when in not_annotated state");
          }

          @Override
          public boolean isDeprecated() {
            throw new IllegalArgumentException(
                "attempt to call gettext on comment when in not_annotated state");
          }
        },
        MatchType.NOT_ANNOTATED);
  }

  public void writeTo(com.google.protobuf.CodedOutputStream output) throws java.io.IOException {
    for (int i = 0; i < classDiff_.size(); i++) {
      output.writeMessage(1, classDiff_.get(i));
    }
    unknownFields.writeTo(output);
  }

  public java.net.URL toJavaURL() {
    try {
      return new java.net.URL(toString());
    } catch (MalformedURLException e) {
      throw new IllegalStateException(e.getMessage(), e);
    }
  }

  public int getSourcePos(int i) {
    throw new IllegalArgumentException(
        "attempt to call gettext on comment when in not_annotated state");
  }

  public void writeTo(com.google.protobuf.CodedOutputStream output) throws java.io.IOException {
    if (!getClassNameBytes().isEmpty()) {
      com.google.protobuf.GeneratedMessageV3.writeString(output, 1, className_);
    }
    for (int i = 0; i < member_.size(); i++) {
      output.writeMessage(2, member_.get(i));
    }
    unknownFields.writeTo(output);
  }

  public String toString() {
    StringBuilder builder = new StringBuilder("costs:\n");
    builder.append("formals=").append(formals).append("\n");
    builder.append("actuals=").append(actuals).append("\n");
    builder.append("costmatrix=\n");
    builder.append(String.format("%20s", ""));
    for (int j = 0; j < costMatrix[0].length; j++) {
      builder.append(String.format("%20s", actuals.get(j).name()));
    }
    builder.append("\n");
    for (int i = 0; i < costMatrix.length; i++) {
      builder.append(String.format("%20s", formals.get(i).name()));
      for (int j = 0; j < costMatrix[i].length; j++) {
        builder.append(String.format("%20.1f", costMatrix[i][j]));
      }
      builder.append("\n");
    }
    return builder.toString();
  }

  private static java.util.logging.Level toJdkLevel(Level level) {
    if (level == Level.ALL) {
      return java.util.logging.Level.ALL;
    }
    if (level == Level.TRACE) {
      return java.util.logging.Level.FINER;
    }
    if (level == Level.DEBUG) {
      return java.util.logging.Level.FINE;
    }
    if (level == Level.INFO) {
      return java.util.logging.Level.INFO;
    }
    if (level == Level.WARN) {
      return java.util.logging.Level.WARNING;
    }
    if (level == Level.ERROR) {
      return java.util.logging.Level.SEVERE;
    }
    return java.util.logging.Level.OFF;
  }

  public static String toTable(List<String> header, List<List<String>> table) {
    int totalWidth = 0;
    int[] widths = new int[header.size()];
    int maxwidth = 70;
    int maxcountbefore = 0;
    for (int j = 0; j < widths.length; j++) {
      widths[j] = Math.max(widths[j], header.get(j).length());
    }
    for (List<String> row : table) {
      int countbefore = 0;
      for (int j = 0; j < widths.length; j++) {
        widths[j] = Math.max(widths[j], row.get(j).length());
        totalWidth = (totalWidth + widths[j]) > maxwidth ? maxwidth : (totalWidth + widths[j]);
        if (j < widths.length - 1) {
          countbefore = countbefore + widths[j];
        }
      }
      maxcountbefore = Math.max(countbefore, maxcountbefore);
    }
    widths[widths.length - 1] = Math.min(widths[widths.length - 1], maxwidth - maxcountbefore);
    StringBuilder buf = new StringBuilder();
    buf.append("+");
    for (int j = 0; j < widths.length; j++) {
      for (int k = 0; k < widths[j] + 2; k++) {
        buf.append("-");
      }
      buf.append("+");
    }
    buf.append("\n");
    buf.append("|");
    for (int j = 0; j < widths.length; j++) {
      String cell = header.get(j);
      buf.append(" ");
      buf.append(cell);
      int pad = widths[j] - cell.length();
      if (pad > 0) {
        for (int k = 0; k < pad; k++) {
          buf.append(" ");
        }
      }
      buf.append(" |");
    }
    buf.append("\n");
    buf.append("+");
    for (int j = 0; j < widths.length; j++) {
      for (int k = 0; k < widths[j] + 2; k++) {
        buf.append("-");
      }
      buf.append("+");
    }
    buf.append("\n");
    for (List<String> row : table) {
      StringBuffer rowbuf = new StringBuffer();
      rowbuf.append("|");
      for (int j = 0; j < widths.length; j++) {
        String cell = row.get(j);
        rowbuf.append(" ");
        int remaing = cell.length();
        while (remaing > 0) {
          if (rowbuf.length() >= totalWidth) {
            buf.append(rowbuf.toString());
            rowbuf = new StringBuffer();
          }
          rowbuf.append(cell.substring(cell.length() - remaing, cell.length() - remaing + 1));
          remaing--;
        }
        int pad = widths[j] - cell.length();
        if (pad > 0) {
          for (int k = 0; k < pad; k++) {
            rowbuf.append(" ");
          }
        }
        rowbuf.append(" |");
      }
      buf.append(rowbuf).append("\n");
    }
    buf.append("+");
    for (int j = 0; j < widths.length; j++) {
      for (int k = 0; k < widths[j] + 2; k++) {
        buf.append("-");
      }
      buf.append("+");
    }
    buf.append("\n");
    return buf.toString();
  }

  public static String toList(List<List<String>> table) {
    int[] widths = new int[table.get(0).size()];
    for (int j = 0; j < widths.length; j++) {
      for (List<String> row : table) {
        widths[j] = Math.max(widths[j], row.get(j).length());
      }
    }
    StringBuilder buf = new StringBuilder();
    for (List<String> row : table) {
      if (buf.length() > 0) {
        buf.append("\n");
      }
      for (int j = 0; j < widths.length; j++) {
        if (j > 0) {
          buf.append(" - ");
        }
        String value = row.get(j);
        buf.append(value);
        if (j < widths.length - 1) {
          int pad = widths[j] - value.length();
          if (pad > 0) {
            for (int k = 0; k < pad; k++) {
              buf.append(" ");
            }
          }
        }
      }
    }
    return buf.toString();
  }
}
