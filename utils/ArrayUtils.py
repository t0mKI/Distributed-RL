def find_sub_array(arr:[], subarr:[]):
    lim=len(arr)-len(subarr)

    for i in range(lim+1):
        match=True
        for j in range(len(subarr)):
            match = match and (arr[i+j]==subarr[j])
            if not match:
                break
        if match:
            return i
    return -1




