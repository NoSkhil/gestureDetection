def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        numset = {}
        freq = [[] for i in range(0,len(nums)+1)]
        
        for num in nums:
            numset[num] = 1 + numset.get(num,0)
        for n,i in numset.items():
            freq[i].append(n)

        res = []
        for index in range(len(freq)-1,-1,-1):
            for n in freq[index]:
                res.append(n)
                if len(res) == k:
                    return res
print("test")


        
