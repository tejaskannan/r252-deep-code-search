public class Results {

  static void crossStreams(final InputStream in, final OutputStream out) throws IOException {
    final byte[] buffer = new byte[bufferSize];
    int len;
    while ((len = in.read(buffer)) != -1) {
      out.write(buffer, 0, len);
    }
  }

  private static void appendWhitespaceIfBr(Element element, StringBuilder accum) {
    if (element.tag.getName().equals("br") && !TextNode.lastCharIsWhitespace(accum))
      accum.append(" ");
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

  private static void writePost(
      final Connection.Request req, final OutputStream outputStream, final String bound)
      throws IOException {
    final Collection<Connection.KeyVal> data = req.data();
    final BufferedWriter w =
        new BufferedWriter(new OutputStreamWriter(outputStream, req.postDataCharset()));
    if (bound != null) {
      for (Connection.KeyVal keyVal : data) {
        w.write("--");
        w.write(bound);
        w.write("\r\n");
        w.write("content-disposition: form-data; name=\"");
        w.write(encodeMimeName(keyVal.key()));
        w.write("\"");
        if (keyVal.hasInputStream()) {
          w.write("; filename=\"");
          w.write(encodeMimeName(keyVal.value()));
          w.write("\"\r\ncontent-type: ");
          w.write(keyVal.contentType() != null ? keyVal.contentType() : DefaultUploadType);
          w.write("\r\n\r\n");
          w.flush();
          DataUtil.crossStreams(keyVal.inputStream(), outputStream);
          outputStream.flush();
        } else {
          w.write("\r\n\r\n");
          w.write(keyVal.value());
        }
        w.write("\r\n");
      }
      w.write("--");
      w.write(bound);
      w.write("--");
    } else if (req.requestBody() != null) {
      w.write(req.requestBody());
    } else {
      boolean first = true;
      for (Connection.KeyVal keyVal : data) {
        if (!first) w.append('&');
        else first = false;
        w.write(URLEncoder.encode(keyVal.key(), req.postDataCharset()));
        w.write('=');
        w.write(URLEncoder.encode(keyVal.value(), req.postDataCharset()));
      }
    }
    w.close();
  }

  public Element appendElement(String tagName) {
    Element child = new Element(Tag.valueOf(tagName, NodeUtils.parser(this).settings()), baseUri());
    appendChild(child);
    return child;
  }

  @Override
  Token reset() {
    reset(name);
    pubSysKey = null;
    reset(publicIdentifier);
    reset(systemIdentifier);
    forceQuirks = false;
    return this;
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
}
