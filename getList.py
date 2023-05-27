def getList() -> list[str]:
  mylist = !ls -m
  elements = list()
  for file in mylist:
    sublist = [x for x in file.split(",")]
    strip_sublist = [y.strip() for y in sublist] #remove whitespaces
    for e in strip_sublist:
      elements.append(e)
  return [i for i in elements if i != ''] #return list of nonempty elements