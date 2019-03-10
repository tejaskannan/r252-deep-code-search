public class Results {

  private String generateMethodArguments(Method method) {
    Class<?>[] pts = method.getParameterTypes();
    return IntStream.range(0, pts.length)
        .mapToObj(i -> String.format(CODE_METHOD_ARGUMENT, pts[i].getCanonicalName(), i))
        .collect(Collectors.joining(", "));
  }

  public static <T extends ServiceDefinition> void build(T sd, final Class<?> interfaceClass) {
    sd.setCanonicalName(interfaceClass.getCanonicalName());
    sd.setCodeSource(ClassUtils.getCodeSource(interfaceClass));
    TypeDefinitionBuilder builder = new TypeDefinitionBuilder();
    List<Method> methods = ClassUtils.getPublicNonStaticMethods(interfaceClass);
    for (Method method : methods) {
      MethodDefinition md = new MethodDefinition();
      md.setName(method.getName());
      Class<?>[] paramTypes = method.getParameterTypes();
      Type[] genericParamTypes = method.getGenericParameterTypes();
      String[] parameterTypes = new String[paramTypes.length];
      for (int i = 0; i < paramTypes.length; i++) {
        TypeDefinition td = builder.build(genericParamTypes[i], paramTypes[i]);
        parameterTypes[i] = td.getType();
      }
      md.setParameterTypes(parameterTypes);
      TypeDefinition td = builder.build(method.getGenericReturnType(), method.getReturnType());
      md.setReturnType(td.getType());
      sd.getMethods().add(md);
    }
    sd.setTypes(builder.getTypeDefinitions());
  }

  private String generateInvocationArgumentNullCheck(Method method) {
    Class<?>[] pts = method.getParameterTypes();
    return IntStream.range(0, pts.length)
        .filter(i -> CLASSNAME_INVOCATION.equals(pts[i].getName()))
        .mapToObj(i -> String.format(CODE_INVOCATION_ARGUMENT_NULL_CHECK, i, i))
        .findFirst()
        .orElse("");
  }

  private String generateUrlAssignmentIndirectly(Method method) {
    Class<?>[] pts = method.getParameterTypes();
    for (int i = 0; i < pts.length; ++i) {
      for (Method m : pts[i].getMethods()) {
        String name = m.getName();
        if ((name.startsWith("get") || name.length() > 3)
            && Modifier.isPublic(m.getModifiers())
            && !Modifier.isStatic(m.getModifiers())
            && m.getParameterTypes().length == 0
            && m.getReturnType() == URL.class) {
          return generateGetUrlNullCheck(i, pts[i], name);
        }
      }
    }
    throw new IllegalStateException(
        "failed to create adaptive class for interface "
            + type.getName()
            + ": not found url parameter or url attribute in parameters of method "
            + method.getName());
  }

  private void generateConstraintsFromAnnotations(
      Type inferredType,
      @Nullable Symbol decl,
      @Nullable Type declaredType,
      Tree sourceTree,
      ArrayDeque<Integer> argSelector) {
    checkArgument(decl == null || argSelector.isEmpty());
    List<Type> inferredTypeArguments = inferredType.getTypeArguments();
    List<Type> declaredTypeArguments =
        declaredType != null ? declaredType.getTypeArguments() : ImmutableList.of();
    int numberOfTypeArgs = inferredTypeArguments.size();
    for (int i = 0; i < numberOfTypeArgs; i++) {
      argSelector.push(i);
      generateConstraintsFromAnnotations(
          inferredTypeArguments.get(i),
          null,
          i < declaredTypeArguments.size() ? declaredTypeArguments.get(i) : null,
          sourceTree,
          argSelector);
      argSelector.pop();
    }
    Optional<Nullness> fromAnnotations = extractExplicitNullness(declaredType, decl);
    if (!fromAnnotations.isPresent()) {
      fromAnnotations = NullnessAnnotations.fromAnnotationsOn(inferredType);
    }
    if (!fromAnnotations.isPresent()) {
      fromAnnotations = NullnessAnnotations.getUpperBound(declaredType);
    }
    fromAnnotations
        .map(ProperInferenceVar::create)
        .ifPresent(
            annot -> {
              InferenceVariable var =
                  TypeArgInferenceVar.create(ImmutableList.copyOf(argSelector), sourceTree);
              qualifierConstraints.putEdge(var, annot);
              qualifierConstraints.putEdge(annot, var);
            });
  }

  public String generate() {
    if (!hasAdaptiveMethod()) {
      throw new IllegalStateException(
          "no adaptive method exist on extension "
              + type.getName()
              + ", refuse to create the adaptive class!");
    }
    StringBuilder code = new StringBuilder();
    code.append(generatePackageInfo());
    code.append(generateImports());
    code.append(generateClassDeclaration());
    Method[] methods = type.getMethods();
    for (Method method : methods) {
      code.append(generateMethod(method));
    }
    code.append("}");
    if (logger.isDebugEnabled()) {
      logger.debug(code.toString());
    }
    return code.toString();
  }

  private void generateConstraintsForWrite(
      Type lType,
      @Nullable Symbol decl,
      ExpressionTree rVal,
      @Nullable Tree lVal,
      ArrayDeque<Integer> argSelector) {
    checkArgument(decl == null || argSelector.isEmpty());
    List<Type> typeArguments = lType.getTypeArguments();
    for (int i = 0; i < typeArguments.size(); i++) {
      argSelector.push(i);
      generateConstraintsForWrite(typeArguments.get(i), null, rVal, lVal, argSelector);
      argSelector.pop();
    }
    ImmutableList<Integer> argSelectorList = ImmutableList.copyOf(argSelector);
    boolean isBound = false;
    Optional<Nullness> fromAnnotations = extractExplicitNullness(lType, decl);
    if (!fromAnnotations.isPresent()) {
      fromAnnotations = NullnessAnnotations.getUpperBound(lType);
      isBound = true;
    }
    boolean oneSided = isBound || argSelector.isEmpty();
    fromAnnotations
        .map(ProperInferenceVar::create)
        .ifPresent(
            annot -> {
              InferenceVariable var = TypeArgInferenceVar.create(argSelectorList, rVal);
              qualifierConstraints.putEdge(var, annot);
              if (!oneSided) {
                qualifierConstraints.putEdge(annot, var);
              }
            });
    if (lVal != null) {
      qualifierConstraints.putEdge(
          TypeArgInferenceVar.create(argSelectorList, rVal),
          TypeArgInferenceVar.create(argSelectorList, lVal));
    }
  }

  @Override
  Optional<SuggestedFix> generateFix(Tree receiver, Tree argument, VisitorState state) {
    return Optional.empty();
  }

  @Override
  Optional<SuggestedFix> generateFix(Tree receiver, Tree argument, VisitorState state) {
    return Optional.empty();
  }

  private static byte[] decodeTable(String code) {
    int hash = code.hashCode();
    byte[] ret = DECODE_TABLE_MAP.get(hash);
    if (ret == null) {
      if (code.length() < 64) {
        throw new IllegalArgumentException("base64 code length < 64.");
      }
      ret = new byte[128];
      for (int i = 0; i < 128; i++) {
        ret[i] = -1;
      }
      for (int i = 0; i < 64; i++) {
        ret[code.charAt(i)] = (byte) i;
      }
      DECODE_TABLE_MAP.put(hash, ret);
    }
    return ret;
  }
}
