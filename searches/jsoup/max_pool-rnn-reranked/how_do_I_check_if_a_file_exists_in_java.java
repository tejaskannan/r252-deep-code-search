public class Results {

  private void checkCapacity(int minNewSize) {
    Validate.isTrue(minNewSize >= size);
    int curSize = keys.length;
    if (curSize >= minNewSize) return;
    int newSize = curSize >= InitialCapacity ? size * GrowthFactor : InitialCapacity;
    if (minNewSize > newSize) newSize = minNewSize;
    keys = copyOf(keys, newSize);
    vals = copyOf(vals, newSize);
  }

  private static String validateCharset(String cs) {
    if (cs == null || cs.length() == 0) return null;
    cs = cs.trim().replaceAll("[\"\']", "");
    try {
      if (Charset.isSupported(cs)) return cs;
      cs = cs.toUpperCase(Locale.ENGLISH);
      if (Charset.isSupported(cs)) return cs;
    } catch (IllegalCharsetNameException e) {
    }
    return null;
  }

  public Connection.Request postDataCharset(String charset) {
    Validate.notNull(charset, "charset must not be null");
    if (!Charset.isSupported(charset)) throw new IllegalCharsetNameException(charset);
    this.postDataCharset = charset;
    return this;
  }

  private void ensureMetaCharsetElement() {
    if (updateMetaCharset) {
      OutputSettings.Syntax syntax = outputSettings().syntax();
      if (syntax == OutputSettings.Syntax.html) {
        Element metaCharset = select("meta[charset]").first();
        if (metaCharset != null) {
          metaCharset.attr("charset", charset().displayName());
        } else {
          Element head = head();
          if (head != null) {
            head.appendElement("meta").attr("charset", charset().displayName());
          }
        }
        select("meta[name=charset]").remove();
      } else if (syntax == OutputSettings.Syntax.xml) {
        Node node = childNodes().get(0);
        if (node instanceof XmlDeclaration) {
          XmlDeclaration decl = (XmlDeclaration) node;
          if (decl.name().equals("xml")) {
            decl.attr("encoding", charset().displayName());
            final String version = decl.attr("version");
            if (version != null) {
              decl.attr("version", "1.0");
            }
          } else {
            decl = new XmlDeclaration("xml", false);
            decl.attr("version", "1.0");
            decl.attr("encoding", charset().displayName());
            prependChild(decl);
          }
        } else {
          XmlDeclaration decl = new XmlDeclaration("xml", false);
          decl.attr("version", "1.0");
          decl.attr("encoding", charset().displayName());
          prependChild(decl);
        }
      }
    }
  }

  static boolean preserveWhitespace(Node node) {
    if (node instanceof Element) {
      Element el = (Element) node;
      int i = 0;
      do {
        if (el.tag.preserveWhitespace()) return true;
        el = el.parent();
        i++;
      } while (i < 6 && el != null);
    }
    return false;
  }

  private static void appendWhitespaceIfBr(Element element, StringBuilder accum) {
    if (element.tag.getName().equals("br") && !TextNode.lastCharIsWhitespace(accum))
      accum.append(" ");
  }

  void reconstructFormattingElements() {
    Element last = lastFormattingElement();
    if (last == null || onStack(last)) return;
    Element entry = last;
    int size = formattingElements.size();
    int pos = size - 1;
    boolean skip = false;
    while (true) {
      if (pos == 0) {
        skip = true;
        break;
      }
      entry = formattingElements.get(--pos);
      if (entry == null || onStack(entry)) break;
    }
    while (true) {
      if (!skip) entry = formattingElements.get(++pos);
      Validate.notNull(entry);
      skip = false;
      Element newEl = insertStartTag(entry.normalName());
      newEl.attributes().addAll(entry.attributes());
      formattingElements.set(pos, newEl);
      if (pos == size - 1) break;
    }
  }

  protected void runParser() {
    while (true) {
      Token token = tokeniser.read();
      process(token);
      token.reset();
      if (token.type == Token.TokenType.EOF) break;
    }
  }

  private void containsData() {
    tq.consume(":containsdata");
    String searchText = TokenQueue.unescape(tq.chompBalanced('(', ')'));
    Validate.notEmpty(searchText, ":containsdata(text) query must not be empty");
    evals.add(new Evaluator.ContainsData(searchText));
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
}
