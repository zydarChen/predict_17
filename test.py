# -*-coding:utf-8 -*-
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None


class Solution:
    def pre_order_rec(self, root):
        if root == None:
            return
        print root.val,
        self.pre_order_rec(root.left)
        self.pre_order_rec(root.right)

    def in_order_rec(self, root):
        if root == None:
            return
        self.in_order_rec(root.left)
        print root.val,
        self.in_order_rec(root.right)

    def post_order_rec(self, root):
        if root == None:
            return
        self.post_order_rec(root.left)
        self.post_order_rec(root.right)
        print root.val,

node1 = TreeNode(1)
node2 = TreeNode(2)
node3 = TreeNode(3)
node4 = TreeNode(4)
node5 = TreeNode(5)
node6 = TreeNode(6)
node7 = TreeNode(7)

node1.left = node2
node1.right = node3
node2.left = node4
node2.right = node5
node3.left = node6
node3.right = node7

S = Solution()
print S.pre_order_rec(node1)
print S.in_order_rec(node1)
print S.post_order_rec(node1)