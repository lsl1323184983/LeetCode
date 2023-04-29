package leetcode.IntroductionToAlgorithms;

import javax.swing.text.AbstractDocument;
import javax.swing.text.rtf.RTFEditorKit;
import java.security.spec.ECField;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Set;
import java.util.TreeMap;

/**
 * @author : lishulong
 * @date : 16:01  2023/4/24
 * @description :
 * @since JDK 11.0
 */
public class Solution {
    /**
     * 704. 二分查找
     * @param nums
     * @param target
     * @return
     */
    public int search(int[] nums, int target){
        int res = -1;
        int mid;
        if(nums[0]==target){
            return 0;
        }
        for (int i = 0,j=nums.length-1; i <= j;) {
            mid = (i+j)/2;
            if(target == nums[mid]){
                res = mid;
                break;
            }else if(target > nums[mid]){
                i = mid + 1;
            }else {
                j = mid - 1;
            }
        }
        return res;
    }

    /**
     * 278. 第一个错误的版本
     * @param n
     * @return
     */
    public int firstBadVersion(int n) {
        int mid;
        int res = -1;
        if(isBadVersion(1)){
            return 1;
        }

        for (int i = 1,j = n; i <= j;) {
            mid = i+(j-i)/2;
            if(isBadVersion(mid)){
                if(isBadVersion(mid - 1)){
                    j=mid-1;
                }else{
                    res = mid;
                    break;
                }
            }else {
                if(isBadVersion(mid + 1)){
                    res = mid+1;
                    break;
                }else{
                    i=mid+1;
                }
            }
        }
        return res;
    }

    public boolean isBadVersion(int n){
        boolean[] version = new boolean[]{false,false,false,false,true,true};
        return version[n-1];
    }


    /**
     * 35. 搜索插入位置
     * @param nums
     * @param target
     * @return
     */
    public int searchInsert(int[] nums, int target) {
        int left = 0;
        int right = nums.length-1;
        int res = nums.length;
        int mid;
        while (left<= right){
            mid = (right - left) / 2 + left;
            if( target <= nums[mid]){
                res = mid;
                right = mid - 1;
            }else {
                left = mid + 1;
            }
        }
        return res;
    }

    /**
     * 977. 有序数组的平方
     * @param nums
     * @return
     */
    public int[] sortedSquares(int[] nums) {
        int n = nums.length;
        int[] res = new int[n];
        int left=0,right=1;
        for (int i = 0; i < n-1; i++) {
            if(nums[i]<=0){
                left = i;
                right = i+1;
            }
        }
        int count = 0;
        for (; left >= 0 && right < n;) {
            if(Math.abs(nums[left])<=Math.abs(nums[right])){
                res[count++] = nums[left] * nums[left];
                left--;
            }else{
                res[count++] = nums[right] * nums[right];
                right++;
            }
        }
        if(left >= 0){
            for (int i = left; i >=0 ; i--) {
                res[count++] = nums[i] * nums[i];
            }
        }
        if(right < n){
            for (int i = right; i < n; i++) {
                res[count++] = nums[i] * nums[i];
            }
        }
        return res;
    }

    /**
     * 189. 轮转数组
     * @param nums
     * @param k
     */
    public void rotate(int[] nums, int k) {
        int n = nums.length;
        int[] newArr = new int[n];
        for (int i = 0; i < n; i++) {
            newArr[(i+k)%n] = nums[i];
        }
        System.arraycopy(newArr,0,nums,0,n);
    }

    public int minPathSum(int[][] grid) {
        int m = grid.length;
        int n = grid[0].length;
        int[][] res = new int[m][n];
        res[0][0] = grid[0][0];
        for (int i = 1; i < m; i++) {
            res[i][0] = grid[i][0]+res[i-1][0];
        }
        for (int i = 1; i < n; i++) {
            res[0][i] = grid[0][i]+res[0][i-1];
        }
        for (int i = 1; i < m; i++) {
            for (int j = 1; j < n; j++) {
                res[i][j] = Math.min(res[i-1][j],res[i][j-1]) + grid[i][j];
            }
        }
        return res[m-1][n-1];
    }

    /**
     * 63. 不同路径 II
     * @param obstacleGrid
     * @return
     */
    public int uniquePathsWithObstacles(int[][] obstacleGrid) {
        int m = obstacleGrid.length;
        int n = obstacleGrid[0].length;
        int[][] res = new int[m][n];
        if(obstacleGrid[0][0] == 0){
            res[0][0] = 1;
        }else{
            res[0][0] = 0;
        }
        for (int i = 1; i < m; i++) {
            if(obstacleGrid[i][0] == 0){
                res[i][0] = res[i-1][0];
            }else{
                res[i][0] = 0;
            }
        }
        for (int i = 1; i < n; i++) {
            if(obstacleGrid[0][i] == 0){
                res[0][i] = res[0][i-1];
            }else{
                res[0][i] = 0;
            }
        }
        for (int i = 1; i < m; i++) {
            for (int j = 1; j < n; j++) {
                if(obstacleGrid[i][j] == 1){
                    res[i][j] = 0;
                }else{
                    res[i][j] = res[i-1][j]+res[i][j-1];
                }
            }
        }
        return res[m-1][n-1];
    }

    /**
     * 283. 移动零
     * @param nums
     */
    public void moveZeroes(int[] nums) {
        int n = nums.length;
        int[] res = new int[n];
        int left=0,right=n-1;
        for (int i = 0; i < n && left <= right; i++) {
            if(nums[i] == 0){
                res[right--] = nums[i];
            }else{
                res[left++] = nums[i];
            }
        }
        System.arraycopy(res,0,nums,0,n);
    }

    /**
     *
     * @param numbers
     * @param target
     * @return
     */
    public int[] twoSum(int[] numbers, int target) {
        int n = numbers.length;
        int[] res = new int[2];
        for (int left = 0,right=n-1; left <= right;) {
            if(numbers[left]+numbers[right] < target){
                left++;
            } else if (numbers[left]+numbers[right] > target) {
                right--;
            }else {
                res[0] = left+1;
                res[1] = right+1;
                return res;
            }
        }
        return res;
    }

    /**
     *
     * @param s
     */
    public void reverseString(char[] s) {
        int n = s.length;
        for (int left = 0,right = n-1; left < right;) {
            char temp = s[left];
            s[left] = s[right];
            s[right] = temp;
            left++;
            right--;
        }
    }


    /**
     * 557. 反转字符串中的单词 III
     * @param s
     * @return
     */
    public String reverseWords(String s) {
        String[] split = s.split("\\s");
        for (int i = 0; i < split.length; i++) {
            char[] chars = split[i].toCharArray();
            reverseString(chars);
            split[i]= new String(chars);
        }
        return String.join(" ",split);
    }

    /**
     *
     * @param head
     * @return
     */
    public ListNode middleNode(ListNode head) {
        int n = 1;
        ListNode left = head;
        ListNode right = head;
        while (right != null) {
            right = right.next;
            n++;
        }
        if(n%2==0){
            n = n / 2;
        }else{
            n = n / 2 + 1;
        }

        while (n > 1){
            left = left.next;
            n--;
        }
        return left;
    }

    /**
     * 19. 删除链表的倒数第 N 个结点
     * @param head
     * @param n
     * @return
     */
    public ListNode removeNthFromEnd(ListNode head, int n) {
        int len = 0;
        ListNode res = new ListNode(-1);
        res.next = head;
        ListNode left = res;
        ListNode right = head;
        while (right != null) {
            right = right.next;
            len++;
        }
        int m = len - n;
        while (m > 0){
            left = left.next;
            m--;
        }
        left.next = left.next.next;
        return res.next;
    }

    /**
     * 70. 爬楼梯
     * @param n
     * @return
     */
    public int climbStairs(int n) {
        int[] res = new int[n+1];
        if(n<=1){
            return 1;
        }
        res[1] = 1;
        res[2] = 2;
        for (int i = 3; i < n+1; i++) {
            res[i] = res[i-1] + res[i-2];
        }
        return res[n];
    }

    /**
     * 509. 斐波那契数
     * @param n
     * @return
     */
    public int fib(int n) {
        if(n==0){
            return 0;
        }
        if(n==1){
            return 1;
        }
        return fib(n-1)+fib(n-2);
    }

    public int tribonacci(int n) {
        int[] res = new int[n+1];
        if(n==0){
            return 0;
        }
        if(n==1||n==2){
            return 1;
        }
        res[0] = 0;
        res[1] = 1;
        res[2] = 1;
        for (int i = 3; i <= n; i++) {
            res[i] = res[i-3]+res[i-2]+res[i-1];
        }
        return res[n];
//        if(n==0){
//            return 0;
//        }
//        if(n==1 || n==2){
//            return 1;
//        }
//        return tribonacci(n-3)+tribonacci(n-2)+tribonacci(n-1);
    }

    /**
     *
     * @param cost
     * @return
     */
    public int minCostClimbingStairs(int[] cost) {
        int n = cost.length;
        if(n==1){
            return 0;
        }
        for (int i = 2; i < n; i++) {
            cost[i] = Math.min(cost[i-1],cost[i-2])+cost[i];
        }
        return Math.min(cost[n-1],cost[n-2]);
    }

    public int rob(int[] nums) {
        int n = nums.length;
        if(n==1){
            return nums[0];
        }
        for (int i = 2; i < n; i++) {
            nums[i] += nums[i-2];
        }
        return Math.max(nums[n-1],nums[n-2]);
    }

    /**
     * 3. 无重复字符的最长子串
     * @param s
     * @return
     */
    public int lengthOfLongestSubstring(String s) {
        Set<Character> occ = new HashSet<>();
        int n = s.length();
        int rk = 0, res = 0;
        for (int i = 0; i < n; i++) {
            if(i!=0){
                occ.remove(s.charAt(i-1));
            }
            while (rk <n&& !occ.contains(s.charAt(rk))){
                occ.add(s.charAt(rk));
                rk++;
            }
            res = Math.max(res,rk-i);
        }
        return res;
    }

    /**
     * 567. 字符串的排列
     * @param s1
     * @param s2
     * @return
     */
    public boolean checkInclusion(String s1, String s2) {
        int n = s1.length();
        int m = s2.length();
        if(n > m){
            return false;
        }
        int[] cnt1 = new int[26];
        int[] cnt2 = new int[26];
        for (int i = 0; i < n; i++) {
            cnt1[s1.charAt(i) - 'a']++;
            cnt2[s2.charAt(i) - 'a']++;
        }
        if(Arrays.equals(cnt1,cnt2)){
            return true;
        }
        for (int i = n; i < m; i++) {
            cnt2[s2.charAt(i) - 'a']++;
            cnt2[s2.charAt(i-n) - 'a']--;
            if(Arrays.equals(cnt1,cnt2)){
                return true;
            }
        }
        return false;
    }
}

class ListNode {
    int val;
    ListNode next;
    ListNode() {}
    ListNode(int val) { this.val = val; }
    ListNode(int val, ListNode next) { this.val = val; this.next = next;}
}
