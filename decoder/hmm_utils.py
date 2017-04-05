def convert_to_dot(filename):
    """Converts the transition matrix shown in a csv file to a
    dot formatted graph.
    """
    csv_file = open(filename)
    lines = csv_file.readlines()
    header = lines[0].split('\t')[1:]  # header for the second one
    graph_string = ""
    for line in lines[1:]:
        feats = line.split('\t')
        domain = feats[0]
        for i in range(1, len(feats)):
            if feats[i].strip() == "1":
                graph_string += domain + " -> " + \
                    header[i-1].strip().strip("\n") + ";\n"
    file.close()
    return graph_string
