public class Results {

  public Element val(String value) {
    if (tagName().equals("textarea")) text(value);
    else attr("value", value);
    return this;
  }

  public String val() {
    if (tagName().equals("textarea")) return text();
    else return attr("value");
  }

  int codepointForName(final String name) {
    int index = Arrays.binarySearch(nameKeys, name);
    return index >= 0 ? codeVals[index] : empty;
  }

  public Parser setTreeBuilder(TreeBuilder treeBuilder) {
    this.treeBuilder = treeBuilder;
    treeBuilder.parser = this;
    return this;
  }

  public void remove(String key) {
    int i = indexOfKey(key);
    if (i != NotFound) remove(i);
  }

  private void remove(int index) {
    Validate.isFalse(index >= size);
    int shifted = size - index - 1;
    if (shifted > 0) {
      System.arraycopy(keys, index + 1, keys, index, shifted);
      System.arraycopy(vals, index + 1, vals, index, shifted);
    }
    size--;
    keys[size] = null;
    vals[size] = null;
  }

  public XmlDeclaration(String name, boolean isProcessingInstruction) {
    Validate.notNull(name);
    value = name;
    this.isProcessingInstruction = isProcessingInstruction;
  }

  protected void removeRange(int fromIndex, int toIndex) {
    onContentsChanged();
    super.removeRange(fromIndex, toIndex);
  }

  void putIgnoreCase(String key, String value) {
    int i = indexOfKeyIgnoreCase(key);
    if (i != NotFound) {
      vals[i] = value;
      if (!keys[i].equals(key)) keys[i] = key;
    } else add(key, value);
  }

  public List<String> headers(String name) {
    Validate.notEmpty(name);
    return getHeadersCaseInsensitive(name);
  }
}
