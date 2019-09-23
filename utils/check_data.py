def check_data(a:list, b:list):
    if len(a) != len(b):
        print('Not the same length!')
        return False
    length = len(a)
    ret = True
    for i in range(length):
        # check_ret = check_src_seqs(a[i]['src_seqs'], b[i]['context_arr'])
        check_trg_seqs(a[i]['trg_seqs'], b[i]['response'])
        check_trg_seqs(a[i]['sketch_seqs'], b[i]['sketch_response'])
        # ret = ret and check_ret
    # print('check for src_seqs: {}'.format(ret))

def check_trg_seqs(a_item, b_item):
    return a_item == b_item

def check_src_seqs(a_item, b_item):
    if len(a_item) != len(b_item):
        return False
    a_item_tup = [tuple(item) for item in a_item]
    b_item_tup = [tuple(item) for item in b_item]
    a_item_tup.sort()
    b_item_tup.sort()
    ret = True
    for i in range(len(a_item_tup)):
        ret = ret and (a_item_tup[i] == b_item_tup[i])
    return ret


