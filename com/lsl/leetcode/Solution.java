package com.lsl.leetcode;

/**
 * @author : lishulong
 * @date : 11:29  2023/3/6
 * @description :
 * @since JDK 11.0
 */
public class Solution {
    /**
     * 62. 不同路径
     * @param m
     * @param n
     * @return
     */
    public int uniquePaths(int m, int n) {
        int[][] path = new int[m][n];
        for (int i = 0; i < m; i++) {
            path[i][0] = 1;
        }
        for (int i = 0; i < n; i++) {
            path[0][i] = 1;
        }
        for (int i = 1; i < m; i++) {
            for (int j = 1; j < n; j++) {
                path[i][j] = path[i-1][j] + path[i][j-1];
            }
        }
        return path[m-1][n-1];
    }

    /**
     * 44. 通配符匹配
     * @param s
     * @param p
     * @return
     */
    public boolean isMatch(String s, String p) {
        int m = s.length(),n = p.length();
        boolean[][] dp = new boolean[m+1][n+1];
        dp[0][0] = true;
        for (int i = 1; i <= n; i++) {
            if(p.charAt(i-1) == '*'){
                dp[0][i] = true;
            }else {
                break;
            }
        }
        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                if(p.charAt(j-1) == '*'){
                    dp[i][j] = dp[i][j-1] || dp[i-1][j];
                } else if (p.charAt(j-1) == '?' || s.charAt(i-1) == p.charAt(j-1)) {
                    dp[i][j] = dp[i-1][j-1];
                }
            }
        }
        return dp[m][n];
    }

    /**
     * 跳跃游戏 II
     * @param nums
     * @return
     */
    public int jump(int[] nums) {
        int length = nums.length;
        int end = 0;
        int maxPosition = 0;
        int steps = 0;
        for (int i = 0; i < length-1; i++) {
            maxPosition = Math.max(maxPosition,i+nums[i]);
            if(i==end){
                end = maxPosition;
                steps++;
            }
        }
        return steps;
    }

    /**
     * 53. 最大子数组和
     * @param nums
     * @return
     */
    public int maxSubArray(int[] nums) {
        int n = nums.length;
        int[] res = new int[n];
        res[0] = nums[0];
        int maxValue = res[0];
        for (int i = 1; i < n; i++) {
            res[i] = Math.max(nums[i],nums[i]+res[i-1]);
            maxValue = Math.max(maxValue,res[i]);
        }
        return maxValue;
    }

    public class Status{
        public int lSum, rSum, mSum, iSum;

        public Status(int lSum, int rSum, int mSum, int iSum){
            this.lSum = lSum;
            this.rSum = rSum;
            this.mSum = mSum;
            this.iSum = iSum;
        }
    }

    /**
     * 53. 最大子数组和(分治法)
     * @param nums
     * @return
     */
    public int maxSubArrayDivideAndConquer(int[] nums) {
        return  getInfo(nums,0,nums.length-1).mSum;
    }

    public Status getInfo(int[] a, int l, int r){
        if(l==r){
            return new Status(a[l],a[l],a[l],a[l]);
        }
        int m = (l+r) >> 1;
        Status lSub = getInfo(a,l,m);
        Status rSub = getInfo(a,m+1,r);
        return pushUp(lSub, rSub);
    }

    public Status pushUp(Status l, Status r) {
        int iSum = l.iSum + r.iSum;
        int lSum = Math.max(l.lSum, l.iSum + r.lSum);
        int rSum = Math.max(r.rSum, r.iSum + l.rSum);
        int mSum = Math.max(Math.max(l.mSum, r.mSum), l.rSum + r.lSum);
        return new Status(lSum, rSum, mSum, iSum);
    }

    /**
     * 55. 跳跃游戏
     * @param nums
     * @return
     */
    public boolean canJump(int[] nums) {
        int n = nums.length;
        int rightmost = 0;
        for (int i = 0; i < n; i++) {
            if(i<=rightmost){
                rightmost =Math.max(nums[i]+i,rightmost);
                if(rightmost >= n-1){
                    return true;
                }
            }
        }
        return false;
    }

}
