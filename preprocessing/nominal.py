class NominalEncoder:
    def __init__(self, table):
        self.table = table

    def encode(self, elements):
        results = [self.table[element] for element in elements]
        return results
    
def create_onehot_table(elements) -> dict:
    # get element set
    element_set = set(elements)
    # create table
    table = dict()
    for i, element in enumerate(element_set):
        value = [0 for _ in range(len(element_set))]
        value[i] = 1
        table[element] = value
    return table

def create_prob_table(elements) -> dict:
    elements = list(elements)
    # get element set
    element_set = set(elements)
    # create table
    table = dict()
    for element in element_set:
        table[element] = elements.count(element) / len(elements)
    return table


if __name__=="__main__":
    elements = ["a", "a", "a", "b", "c"]

    table = create_prob_table(elements)
    encoder = NominalEncoder(table)
    result = encoder.encode(elements)
    print(result)