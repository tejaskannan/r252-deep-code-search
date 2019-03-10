public class Results {

  private static void appendWhitespaceIfBr(Element element, StringBuilder accum) {
    if (element.tag.getName().equals("br") && !TextNode.lastCharIsWhitespace(accum))
      accum.append(" ");
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

  public Elements parents() {
    HashSet<Element> combo = new LinkedHashSet<>();
    for (Element e : this) {
      combo.addAll(e.parents());
    }
    return new Elements(combo);
  }

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

  public Element appendElement(String tagName) {
    Element child = new Element(Tag.valueOf(tagName, NodeUtils.parser(this).settings()), baseUri());
    appendChild(child);
    return child;
  }

  private static BomCharset detectCharsetFromBom(final ByteBuffer byteData) {
    final Buffer buffer = byteData;
    buffer.mark();
    byte[] bom = new byte[4];
    if (byteData.remaining() >= bom.length) {
      byteData.get(bom);
      buffer.rewind();
    }
    if (bom[0] == 00 && bom[1] == 00 && bom[2] == (byte) fe && bom[3] == (byte) ff
        || bom[0] == (byte) ff && bom[1] == (byte) fe && bom[2] == 00 && bom[3] == 00) {
      return new BomCharset("utf-32", false);
    } else if (bom[0] == (byte) fe && bom[1] == (byte) ff
        || bom[0] == (byte) ff && bom[1] == (byte) fe) {
      return new BomCharset("utf-16", false);
    } else if (bom[0] == (byte) ef && bom[1] == (byte) bb && bom[2] == (byte) bf) {
      return new BomCharset("utf-8", true);
    }
    return null;
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

  @Override
  Tag reset() {
    tagName = null;
    normalName = null;
    pendingAttributeName = null;
    reset(pendingAttributeValue);
    pendingAttributeValueS = null;
    hasEmptyAttributeValue = false;
    hasPendingAttributeValue = false;
    selfClosing = false;
    attributes = null;
    return this;
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
}
