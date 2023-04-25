package leetcode.IntroductionToAlgorithms;

import javax.swing.text.AbstractDocument;
import javax.swing.text.rtf.RTFEditorKit;
import java.security.spec.ECField;
import java.util.Arrays;
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
}
