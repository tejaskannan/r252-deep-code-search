public class Results {

  public int hashCode() {
    int result = tagName.hashCode();
    result = 31 * result + (isBlock ? 1 : 0);
    result = 31 * result + (formatAsBlock ? 1 : 0);
    result = 31 * result + (canContainInline ? 1 : 0);
    result = 31 * result + (empty ? 1 : 0);
    result = 31 * result + (selfClosing ? 1 : 0);
    result = 31 * result + (preserveWhitespace ? 1 : 0);
    result = 31 * result + (formList ? 1 : 0);
    result = 31 * result + (formSubmit ? 1 : 0);
    return result;
  }

  public Elements getElementsByAttributeValueMatching(String key, Pattern pattern) {
    return Collector.collect(new Evaluator.AttributeWithValueMatching(key, pattern), this);
  }

  void outerHtmlHead(Appendable accum, int depth, Document.OutputSettings out) throws IOException {
    accum.append("<").append(isProcessingInstruction ? "!" : "?").append(coreValue());
    getWholeDeclaration(accum, out);
    accum.append(isProcessingInstruction ? "!" : "?").append(">");
  }

  public Elements getElementsByAttributeValueMatching(String key, String regex) {
    Pattern pattern;
    try {
      pattern = Pattern.compile(regex);
    } catch (PatternSyntaxException e) {
      throw new IllegalArgumentException("pattern syntax error: " + regex, e);
    }
    return getElementsByAttributeValueMatching(key, pattern);
  }

  public String data() {
    StringBuilder sb = StringUtil.borrowBuilder();
    for (Node childNode : childNodes) {
      if (childNode instanceof DataNode) {
        DataNode data = (DataNode) childNode;
        sb.append(data.getWholeData());
      } else if (childNode instanceof Comment) {
        Comment comment = (Comment) childNode;
        sb.append(comment.getData());
      } else if (childNode instanceof Element) {
        Element element = (Element) childNode;
        String elementData = element.data();
        sb.append(elementData);
      } else if (childNode instanceof CDataNode) {
        CDataNode cDataNode = (CDataNode) childNode;
        sb.append(cDataNode.getWholeText());
      }
    }
    return StringUtil.releaseBuilder(sb);
  }

  public Connection data(String... keyvals) {
    Validate.notNull(keyvals, "data key value pairs must not be null");
    Validate.isTrue(keyvals.length % 2 == 0, "must supply an even number of key value pairs");
    for (int i = 0; i < keyvals.length; i += 2) {
      String key = keyvals[i];
      String value = keyvals[i + 1];
      Validate.notEmpty(key, "data key must not be empty");
      Validate.notNull(value, "data value must not be null");
      req.data(KeyVal.create(key, value));
    }
    return this;
  }

  public int hashCode() {
    final int prime = 31;
    int result = 1;
    result = prime * result + ((value == null) ? 0 : value.hashCode());
    return result;
  }

  private void clearStackToContext(String... nodeNames) {
    for (int pos = stack.size() - 1; pos >= 0; pos--) {
      Element next = stack.get(pos);
      if (StringUtil.in(next.normalName(), nodeNames) || next.normalName().equals("html")) break;
      else stack.remove(pos);
    }
  }

  @Override
  void outerHtmlHead(Appendable accum, int depth, Document.OutputSettings out) throws IOException {
    accum.append("<![cdata[").append(getWholeText());
  }

  public String attr(String key) {
    Validate.notNull(key);
    if (!hasAttributes()) {
      return key.equals(nodeName()) ? (String) value : EmptyString;
    }
    return super.attr(key);
  }
}
