public class Results {

  public Element prependElement(String tagName) {
    Element child = new Element(Tag.valueOf(tagName, NodeUtils.parser(this).settings()), baseUri());
    prependChild(child);
    return child;
  }

  void generateImpliedEndTags(String excludeTag) {
    while ((excludeTag != null && !currentElement().normalName().equals(excludeTag))
        && inSorted(currentElement().normalName(), TagSearchEndTags)) pop();
  }

  public T cookie(String name, String value) {
    Validate.notEmpty(name, "cookie name must not be empty");
    Validate.notNull(value, "cookie value must not be null");
    cookies.put(name, value);
    return (T) this;
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

  void outerHtmlHead(Appendable accum, int depth, Document.OutputSettings out) throws IOException {
    accum.append("<").append(isProcessingInstruction ? "!" : "?").append(coreValue());
    getWholeDeclaration(accum, out);
    accum.append(isProcessingInstruction ? "!" : "?").append(">");
  }

  final void html(final Appendable accum, final Document.OutputSettings out) throws IOException {
    final int sz = size;
    for (int i = 0; i < sz; i++) {
      final String key = keys[i];
      final String val = vals[i];
      accum.append(' ').append(key);
      if (!Attribute.shouldCollapseAttribute(key, val, out)) {
        accum.append("=\"");
        Entities.escape(accum, val == null ? EmptyString : val, out, true, false, false);
        accum.append('\"');
      }
    }
  }

  void popStackToBefore(String elName) {
    for (int pos = stack.size() - 1; pos >= 0; pos--) {
      Element next = stack.get(pos);
      if (next.normalName().equals(elName)) {
        break;
      } else {
        stack.remove(pos);
      }
    }
  }

  public Node attr(String attributeKey, String attributeValue) {
    attributeKey = NodeUtils.parser(this).settings().normalizeAttribute(attributeKey);
    attributes().putIgnoreCase(attributeKey, attributeValue);
    return this;
  }

  private void clearStackToContext(String... nodeNames) {
    for (int pos = stack.size() - 1; pos >= 0; pos--) {
      Element next = stack.get(pos);
      if (StringUtil.in(next.normalName(), nodeNames) || next.normalName().equals("html")) break;
      else stack.remove(pos);
    }
  }

  void popStackToClose(String elName) {
    for (int pos = stack.size() - 1; pos >= 0; pos--) {
      Element next = stack.get(pos);
      stack.remove(pos);
      if (next.normalName().equals(elName)) break;
    }
  }
}
