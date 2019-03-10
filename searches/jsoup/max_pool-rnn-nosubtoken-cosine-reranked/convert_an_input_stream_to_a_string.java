public class Results {

  public static String unescape(String in) {
    StringBuilder out = StringUtil.borrowBuilder();
    char last = 0;
    for (char c : in.toCharArray()) {
      if (c == ESC) {
        if (last != 0 && last == ESC) out.append(c);
      } else out.append(c);
      last = c;
    }
    return StringUtil.releaseBuilder(out);
  }

  public static void appendNormalisedWhitespace(
      StringBuilder accum, String string, boolean stripLeading) {
    boolean lastWasWhite = false;
    boolean reachedNonWhite = false;
    int len = string.length();
    int c;
    for (int i = 0; i < len; i += Character.charCount(c)) {
      c = string.codePointAt(i);
      if (isActuallyWhitespace(c)) {
        if ((stripLeading && !reachedNonWhite) || lastWasWhite) continue;
        accum.append(' ');
        lastWasWhite = true;
      } else if (!isInvisibleChar(c)) {
        accum.appendCodePoint(c);
        lastWasWhite = false;
        reachedNonWhite = true;
      }
    }
  }

  private static String fixHeaderEncoding(String val) {
    try {
      byte[] bytes = val.getBytes("iso-8859-1");
      if (!looksLikeUtf8(bytes)) return val;
      return new String(bytes, "utf-8");
    } catch (UnsupportedEncodingException e) {
      return val;
    }
  }

  static URL encodeUrl(URL u) {
    try {
      String urlS = u.toExternalForm();
      urlS = urlS.replaceAll(" ", "%20");
      final URI uri = new URI(urlS);
      return new URL(uri.toASCIIString());
    } catch (Exception e) {
      return u;
    }
  }

  private static void appendNormalisedText(StringBuilder accum, TextNode textNode) {
    String text = textNode.getWholeText();
    if (preserveWhitespace(textNode.parentNode) || textNode instanceof CDataNode)
      accum.append(text);
    else StringUtil.appendNormalisedWhitespace(accum, text, TextNode.lastCharIsWhitespace(accum));
  }

  public static String join(Iterator strings, String sep) {
    if (!strings.hasNext()) return "";
    String start = strings.next().toString();
    if (!strings.hasNext()) return start;
    StringBuilder sb = StringUtil.borrowBuilder().append(start);
    while (strings.hasNext()) {
      sb.append(sep);
      sb.append(strings.next());
    }
    return StringUtil.releaseBuilder(sb);
  }

  public static String escape(String string, Document.OutputSettings out) {
    if (string == null) return "";
    StringBuilder accum = StringUtil.borrowBuilder();
    try {
      escape(accum, string, out, false, false, false);
    } catch (IOException e) {
      throw new SerializationException(e);
    }
    return StringUtil.releaseBuilder(accum);
  }

  public String cssSelector() {
    if (id().length() > 0) return "#" + id();
    String tagName = tagName().replace(':', '|');
    StringBuilder selector = new StringBuilder(tagName);
    String classes = StringUtil.join(classNames(), ".");
    if (classes.length() > 0) selector.append('.').append(classes);
    if (parent() == null || parent() instanceof Document) return selector.toString();
    selector.insert(0, " > ");
    if (parent().select(selector.toString()).size() > 1)
      selector.append(String.format(":nth-child(%d)", elementSiblingIndex() + 1));
    return parent().cssSelector() + selector.toString();
  }

  void resetInsertionMode() {
    boolean last = false;
    for (int pos = stack.size() - 1; pos >= 0; pos--) {
      Element node = stack.get(pos);
      if (pos == 0) {
        last = true;
        node = contextElement;
      }
      String name = node.normalName();
      if ("select".equals(name)) {
        transition(HtmlTreeBuilderState.InSelect);
        break;
      } else if (("td".equals(name) || "th".equals(name) && !last)) {
        transition(HtmlTreeBuilderState.InCell);
        break;
      } else if ("tr".equals(name)) {
        transition(HtmlTreeBuilderState.InRow);
        break;
      } else if ("tbody".equals(name) || "thead".equals(name) || "tfoot".equals(name)) {
        transition(HtmlTreeBuilderState.InTableBody);
        break;
      } else if ("caption".equals(name)) {
        transition(HtmlTreeBuilderState.InCaption);
        break;
      } else if ("colgroup".equals(name)) {
        transition(HtmlTreeBuilderState.InColumnGroup);
        break;
      } else if ("table".equals(name)) {
        transition(HtmlTreeBuilderState.InTable);
        break;
      } else if ("head".equals(name)) {
        transition(HtmlTreeBuilderState.InBody);
        break;
      } else if ("body".equals(name)) {
        transition(HtmlTreeBuilderState.InBody);
        break;
      } else if ("frameset".equals(name)) {
        transition(HtmlTreeBuilderState.InFrameset);
        break;
      } else if ("html".equals(name)) {
        transition(HtmlTreeBuilderState.BeforeHead);
        break;
      } else if (last) {
        transition(HtmlTreeBuilderState.InBody);
        break;
      }
    }
  }

  private static void appendWhitespaceIfBr(Element element, StringBuilder accum) {
    if (element.tag.getName().equals("br") && !TextNode.lastCharIsWhitespace(accum))
      accum.append(" ");
  }
}
