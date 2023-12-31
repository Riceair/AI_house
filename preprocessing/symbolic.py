class SymbolicEncoder:
    def __init__(self, table):
        self.table = table

    def encode(self, elements):
        results = [self.table[element] for element in elements]
        return results
    
def create_onehot_table(elements, target_min=0, target_max=1) -> dict:
    # get element set
    element_set = list(set(elements))
    element_set = sorted(element_set)
    # create table
    table = dict()
    for i, element in enumerate(element_set):
        value = [target_min for _ in range(len(element_set))]
        value[i] = target_max
        table[element] = value
    return table

def create_order_table(elements) -> dict:
    # get element set
    element_set = list(set(elements))
    element_set = sorted(element_set)
    element_count = len(element_set)
    # create table
    table = dict()
    for i, element in enumerate(element_set):
        table[element] = i/element_count
    return table

def create_y_order_table(elements, targets, target_min=0, target_max=1):
    # get element set
    element_set = list(set(elements))
    element_set = sorted(element_set)
    # create table
    table = dict()
    e_count_table = dict()
    # init table
    for element in element_set:
        table[element] = 0
        e_count_table[element] = 0
    # accumulate y values
    for element, target in zip(elements, targets):
        if element != "None":
            table[element] += target
        e_count_table[element] += 1
    
    for key in table:
        table[key] /= e_count_table[key]
    
    # Normalize to target min and max
    original_min = min(table.values())
    original_max = max(table.values())
    for key in table:
        value = table[key]
        table[key] = (value - original_min) / (original_max - original_min) *\
                     (target_max - target_min) + target_min
    return table

def create_prob_table(elements) -> dict:
    elements = list(elements)
    # get element set
    element_set = list(set(elements))
    element_set = sorted(element_set)
    # create table
    table = dict()
    for element in element_set:
        table[element] = elements.count(element) / len(elements)
    return table


if __name__=="__main__":
    elements = ["其他", "商", "商", "工", "None"]
    y_true = [10, 50, 60, 40, 1000]
    table = create_y_order_table(elements, y_true)
    encoder = SymbolicEncoder(table)
    result = encoder.encode(elements)
    print(result)