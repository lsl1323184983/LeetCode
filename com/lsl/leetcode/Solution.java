package com.lsl.leetcode;

import com.sun.management.GcInfo;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

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

    /**
     * 312. 戳气球
     * @param nums
     * @return
     */
    public int[][] rec;
    public int[] val;
    public int maxCoins(int[] nums) {
        int n = nums.length;
        val = new int[n+2];
        for (int i = 1; i <= n; i++) {
            val[i] = nums[i-1];
        }
        val[0] = val[n+1] = 1;
        rec = new int[n+2][n+2];
        for (int i = 0; i <= n+1; i++) {
            Arrays.fill(rec[i],-1);
        }
        return solve(0,n+1);
    }

    public int solve(int left,int right){
        if(left >= right - 1){
            return 0;
        }
        if(rec[left][right] != -1){
            return rec[left][right];
        }
        for (int i = left+1; i < right; i++) {
            int sum = val[left] * val[i] * val[right];
            sum += solve(left,i) + solve(i,right);
            rec[left][right] = Math.max(rec[left][right], sum);
        }
        return rec[left][right];
    }


    public int maxCoins_dp(int[] nums) {
        int n = nums.length;
        int[][] res = new int[n+2][n+2];
        int[] val = new int[n+2];
        val[0] = val[n+1] = 1;
        for (int i = 1; i <= n; i++) {
            val[i] = nums[i-1];
        }
        for (int i = n-1; i >=0 ; i--) {
            for (int j = i+2; j <= n+1 ; j++) {
                for (int k = i+1; k < j; k++) {
                    int sum = val[i] * val[k] * val[j];
                    sum += res[i][k] + res[k][j];
                    res[i][j] = Math.max(res[i][j],sum);
                }
            }
        }
        return res[0][n+1];
    }

    public int maxCoins_dp_2(int[] nums) {
        int n = nums.length;
        int[][] res = new int[n+2][n+2];
        int[] val = new int[n+2];
        val[0] = val[n+1] = 1;
        for (int i = 1; i <= n; i++) {
            val[i] = nums[i-1];
        }
        for (int i = 2; i <= n+1; i++) {
            for (int j = i - 2; j >= 0; j--) {
                for (int k = j + 1; k < i; k++) {
                    int sum = val[i] * val[k] * val[j];
                    sum += res[i][k] + res[k][j];
                    res[i][j] = Math.max(res[i][j],sum);
                }
            }
        }
        return res[n+1][0];
    }

    /**
     * 72. 编辑距离
     * @param word1
     * @param word2
     * @return
     */
    public int minDistance(String word1, String word2) {
        int n = word1.length();
        int m = word2.length();
        if(n*m == 0){
            return n+m;
        }
        int[][] dp = new int[n+1][m+1];
        for (int i = 0; i < n+1; i++) {
            dp[i][0] = i;
        }
        for (int j = 0; j < m+1; j++) {
            dp[0][j] = j;
        }
        for (int i = 1; i < n+1; i++) {
            for (int j = 1; j < m+1; j++) {
                int left = dp[i-1][j]+1;
                int down = dp[i][j-1]+1;
                int left_down = dp[i-1][j-1];
                if(word1.charAt(i-1)!=word2.charAt(j-1)){
                    left_down += 1;
                }
                dp[i][j] = Math.min(left,Math.min(down,left_down));
            }
        }
        return dp[n][m];
    }


    static final int MOD = 1000000007;
    public int numDecodings(String s){
        int n = s.length();
        long a=0,b=1,c=0;
        for (int i = 1; i <= n; i++) {
            c = b * check1digit(s.charAt(i-1)) % MOD;
            if(i>1){
                c = (c + a * check2digits(s.charAt(i-2),s.charAt(i-1))) % MOD;
            }
            a = b;
            b = c;
        }
        return (int) c;
    }

    public int check1digit(char ch) {
        if (ch == '0') {
            return 0;
        }
        return ch == '*' ? 9 : 1;
    }

    public int check2digits(char c0, char c1) {
        if (c0 == '*' && c1 == '*') {
            return 15;
        }
        if (c0 == '*') {
            return c1 <= '6' ? 2 : 1;
        }
        if (c1 == '*') {
            if (c0 == '1') {
                return 9;
            }
            if (c0 == '2') {
                return 6;
            }
            return 0;
        }
        return (c0 != '0' && (c0 - '0') * 10 + (c1 - '0') <= 26) ? 1 : 0;
    }

    /**
     *
     * @param nums
     * @return
     */
    public boolean canPartition(int[] nums) {
        int n = nums.length;
        if(n<2){
            return false;
        }
        int sum = 0,maxNum = 0;
        for (int num : nums) {
            sum = sum + num;
            maxNum = Math.max(num,maxNum);
        }
        if(sum % 2 != 0){
            return false;
        }
        int target = sum / 2;
        if(maxNum > target){
            return false;
        }
        boolean [][]dp = new boolean[n][target+1];
        for (int i = 0; i < n; i++) {
            dp[i][0] = true;
        }
        dp[0][nums[0]] = true;
        for (int i = 1; i < n; i++) {
            int num = nums[i];
            for (int j = 1; j <= target; j++) {
                if(j>=num){
                    dp[i][j] = dp[i-1][j]|dp[i-1][j-num];
                }else{
                    dp[i][j] = dp[i-1][j];
                }
            }
        }
        return dp[n-1][target];
    }

    /**
     * 给定一个整数数组，如果可以将数组划分为两个子集，使得两个子集中的元素之和相等返回true,反之 false
     * @param nums
     * @return
     */
    public boolean canPartition_improvement(int[] nums) {
        int n = nums.length;
        if(n<2){
            return false;
        }
        int sum = 0,maxNum = 0;
        for (int num : nums) {
            sum = sum + num;
            maxNum = Math.max(num,maxNum);
        }
        if(sum % 2 != 0){
            return false;
        }
        int target = sum / 2;
        if(maxNum > target){
            return false;
        }

        boolean []dp = new boolean[target+1];
        dp[0]=true;
        for (int num : nums) {
            for (int i=target;i>=num;i--){
                dp[i] = dp[i] | dp[i-num];
            }
        }
        return dp[target];
    }


    /**
     * 给定一个整数数组，如果可以将数组划分为两个子集，使得两个子集中的元素之和相等返回true,反之 false
     * @param nums
     * @return
     */
    public int lengthOfLIS(int[] nums) {
        int []dp = new int[nums.length];
        dp[0] = 1;
        int maxans = 1;
        for (int i = 1; i < nums.length; i++) {
            dp[i] = 1;
            for (int j = 0; j < i; j++) {
                if(nums[i] > nums[j]){
                    dp[i] = Math.max(dp[i],dp[j]+1);
                }
            }
            maxans = Math.max(maxans, dp[i]);
        }
        return maxans;
    }

    /**
     * 堆排序
     * @param sourceArray
     * @return
     */
    public int[] HeapSort(int[] sourceArray){
        int[] arr = Arrays.copyOf(sourceArray,sourceArray.length);
        int len = arr.length;
        buildMaxHeap(arr, len);
        for (int i = len-1; i > 0 ; i--) {
            swap(arr,0,i);
            len--;
            heapify(arr,0,len);
        }
        return arr;
    }

    public void buildMaxHeap(int[] arr,int len){
        for (int i = (int) Math.floor(len / 2); i >= 0; i--) {
            heapify(arr, i, len);
        }
    }

    public void heapify(int[] arr,int i,int len){
        int left = 2 * i + 1;
        int right = 2 * i + 2;
        int largest = i;

        if(left < len && arr[left] < arr[largest]){
            largest = left;
        }

        if(right < len && arr[right] < arr[largest]){
            largest = right;
        }

        if(largest != i){
            swap(arr, i, largest);
            heapify(arr, largest, len);
        }
    }

    public void swap(int[] arr,int i,int j){
        int temp = arr[i];
        arr[i] = arr[j];
        arr[j] = temp;
    }

    /**
     * Tarjan 算法,用于求解图的连通性问题
     */
    public void tarjan(){
        int length = 5;
        int[] dfn = new int[length];
        int[] low = new int[length];
    }


    /**
     * 15. 三数之和 (排序 + 双指针)
     * 给你一个整数数组 nums ，判断是否存在三元组 [nums[i], nums[j], nums[k]] 满足 i != j、i != k 且 j != k ，同时还满足 nums[i] + nums[j] + nums[k] == 0 。请
     * 你返回所有和为 0 且不重复的三元组。
     * @param nums
     * @return
     */
    public List<List<Integer>> threeSum(int[] nums) {
        Arrays.sort(nums);
        int n = nums.length;
        List<List<Integer>> res = new ArrayList<List<Integer>>();
        for (int first = 0; first < n; first++) {
            if(first > 0 && nums[first] == nums[first-1]){
                continue;
            }
            int third = n-1;
            int target = -nums[first];
            for (int second = first+1; second < n; second++) {
                if(second > first + 1 && nums[second] == nums[second - 1]){
                    continue;
                }
                while (second < third && nums[second] + nums[third] > target){
                    third--;
                }
                if (second == third){
                    break;
                }
                if(nums[second] + nums[third] == target){
                    List<Integer> list = new ArrayList<>();
                    list.add(nums[first]);
                    list.add(nums[second]);
                    list.add(nums[third]);
                    res.add(list);
                }
            }
        }
        return res;
    }
}
