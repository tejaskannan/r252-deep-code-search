public class Results {

  private static ImmutableSet<String> getAnnotationValueAsStrings(Symbol sym, String guardedBy) {
    return sym.getRawAttributes().stream()
        .filter(a -> a.getAnnotationType().asElement().getSimpleName().contentEquals(guardedBy))
        .flatMap(
            a ->
                a.getElementValues().entrySet().stream()
                    .filter(e -> e.getKey().getSimpleName().contentEquals("value"))
                    .map(Map.Entry::getValue)
                    .findFirst()
                    .map(GuardedByUtils::asStrings)
                    .orElse(Stream.empty()))
        .collect(toImmutableSet());
  }

  public Description matchSwitch(SwitchTree tree, VisitorState state) {
    PeekingIterator<JCTree.JCCase> it =
        Iterators.peekingIterator(((JCTree.JCSwitch) tree).cases.iterator());
    while (it.hasNext()) {
      JCTree.JCCase caseTree = it.next();
      if (!it.hasNext()) {
        break;
      }
      JCTree.JCCase next = it.peek();
      if (caseTree.stats.isEmpty()) {
        continue;
      }
      boolean completes = Reachability.canCompleteNormally(getLast(caseTree.stats));
      String comments =
          state
              .getSourceCode()
              .subSequence(caseEndPosition(state, caseTree), next.getStartPosition())
              .toString()
              .trim();
      if (completes && !FALL_THROUGH_PATTERN.matcher(comments).find()) {
        state.reportMatch(
            buildDescription(next)
                .setMessage(
                    "execution may fall through from the previous case; add a `// fall through`"
                        + " comment before this line if it was deliberate")
                .build());
      } else if (!completes && FALL_THROUGH_PATTERN.matcher(comments).find()) {
        state.reportMatch(
            buildDescription(next)
                .setMessage(
                    "switch case has \'fall through\' comment, but execution cannot fall through"
                        + " from the previous case")
                .build());
      }
    }
    return NO_MATCH;
  }

  public static <A extends Annotation> boolean isPresent(Method method, Class<A> annotationClass) {
    Map<ElementType, List<A>> annotationsMap = findAnnotations(method, annotationClass);
    return !annotationsMap.isEmpty();
  }

  private long hash(byte[] digest, int number) {
    return (((long) (digest[3 + number * 4] & 0xff) << 24)
            | ((long) (digest[2 + number * 4] & 0xff) << 16)
            | ((long) (digest[1 + number * 4] & 0xff) << 8)
            | (digest[number * 4] & 0xff))
        & ffffffff;
  }

  Stream<ParameterPair> viablePairs() {
    return formals.stream()
        .flatMap(f -> actuals.stream().map(a -> ParameterPair.create(f, a)))
        .filter(
            p -> costMatrix[p.formal().index()][p.actual().index()] != Double.POSITIVE_INFINITY);
  }

  private Invoker<T> selectForKey(long hash) {
    Map.Entry<Long, Invoker<T>> entry = virtualInvokers.ceilingEntry(hash);
    if (entry == null) {
      entry = virtualInvokers.firstEntry();
    }
    return entry.getValue();
  }

  private static ImmutableList<Commented<ExpressionTree>> noComments(
      List<? extends ExpressionTree> arguments) {
    return arguments.stream()
        .map(a -> Commented.<ExpressionTree>builder().setTree(a).build())
        .collect(toImmutableList());
  }

  public int hashCode() {
    int h = 1;
    h *= 1000003;
    h ^= this.onByDefault ? 1231 : 1237;
    h *= 1000003;
    h ^= this.severity.hashCode();
    return h;
  }

  public int hashCode() {
    int h = 1;
    h *= 1000003;
    h ^= this.matchingNodes.hashCode();
    h *= 1000003;
    h ^= this.matches ? 1231 : 1237;
    return h;
  }

  public int hashCode() {
    int h = 1;
    h *= 1000003;
    h ^= this.methodPath.hashCode();
    return h;
  }
}
