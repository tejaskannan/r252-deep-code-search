public class Results {

  private ConstrainableInputStream(InputStream in, int bufferSize, int maxSize) {
    super(in, bufferSize);
    Validate.isTrue(maxSize >= 0);
    this.maxSize = maxSize;
    remaining = maxSize;
    capped = maxSize != 0;
    startTime = System.nanoTime();
  }

  public byte[] bodyAsBytes() {
    prepareByteData();
    return byteData.array();
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

  public List<DataNode> dataNodes() {
    List<DataNode> dataNodes = new ArrayList<>();
    for (Node node : childNodes) {
      if (node instanceof DataNode) dataNodes.add((DataNode) node);
    }
    return Collections.unmodifiableList(dataNodes);
  }

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
}
