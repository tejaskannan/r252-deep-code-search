public class Results {

  public int hashCode() {
    final int prime = 31;
    int result = 1;
    result = prime * result + ((value == null) ? 0 : value.hashCode());
    return result;
  }

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

  void generateImpliedEndTags(String excludeTag) {
    while ((excludeTag != null && !currentElement().normalName().equals(excludeTag))
        && inSorted(currentElement().normalName(), TagSearchEndTags)) pop();
  }

  void popStackToClose(String elName) {
    for (int pos = stack.size() - 1; pos >= 0; pos--) {
      Element next = stack.get(pos);
      stack.remove(pos);
      if (next.normalName().equals(elName)) break;
    }
  }

  void popStackToClose(String... elNames) {
    for (int pos = stack.size() - 1; pos >= 0; pos--) {
      Element next = stack.get(pos);
      stack.remove(pos);
      if (inSorted(next.normalName(), elNames)) break;
    }
  }

  public Element val(String value) {
    if (tagName().equals("textarea")) text(value);
    else attr("value", value);
    return this;
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

  public Connection cookie(String name, String value) {
    req.cookie(name, value);
    return this;
  }

  private void popStackToClose(Token.EndTag endTag) {
    String elName = settings.normalizeTag(endTag.tagName);
    Element firstFound = null;
    for (int pos = stack.size() - 1; pos >= 0; pos--) {
      Element next = stack.get(pos);
      if (next.nodeName().equals(elName)) {
        firstFound = next;
        break;
      }
    }
    if (firstFound == null) return;
    for (int pos = stack.size() - 1; pos >= 0; pos--) {
      Element next = stack.get(pos);
      stack.remove(pos);
      if (next == firstFound) break;
    }
  }

  public Elements val(String value) {
    for (Element element : this) element.val(value);
    return this;
  }
}
