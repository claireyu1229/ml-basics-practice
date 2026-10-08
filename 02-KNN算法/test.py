from typing import List


# 回溯
class Solution:
    # 递归将结果到树的底部,将结果都存到result列表中去,这里传入的参数都是引用因此
    # 注意保存的结果要创建一个副本
    def backtracking(self, n, k, stat_index: int, path: list, result: list):
        # 递归先写终止条件当path的长度要的k就结束,保存当前叶子节点到result结果
        if len(path) == k:
            result.append(path.copy())
            return
        # 要将每个数作为开头都遍历一遍
        for i in range(stat_index,n - (k - len(path)) + 2):
            path.append(i)
            # 是将当前节点作为父节点将所有子节点符合k的都添加到result
            # 这里传的参数start_index就要改变一下因为时下一层了所以要加1
            self.backtracking(n, k, i + 1, path, result)
            path.pop()

    def combine(self, n: int, k: int) -> List[List[int]]:
        result = []
        self.backtracking(n, k, 1, [], result)
        return result

    # 剪枝


s = Solution()
print(s.combine(4, 2))

