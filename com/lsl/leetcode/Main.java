package com.lsl.leetcode;

import java.util.Scanner;
import java.util.concurrent.SynchronousQueue;

/**
 * @author : lishulong
 * @date : 11:36  2023/3/6
 * @description :
 * @since JDK 11.0
 */
public class Main {
    public static void main(String[] args) {
        Solution solution = new Solution();
        int []nums = new int[]{7,7,7,7,7,7,7};
        System.out.println(solution.lengthOfLIS(nums));
    }
//    public static void main(String[] args) {
//        Scanner sc = new Scanner(System.in);
//        String str = sc.nextLine();
//        int n = str.length();
//        int res = 0;
//        for (int i = 1,j=0; i < n;) {
//            if(str.charAt(i)==str.charAt(j)){
//                res += 1;
//                i += 2;
//                j += 2;
//            }else {
//                i += 1;
//                j += 1;
//            }
//        }
//        System.out.println(res);
//    }

//    public static void main(String[] args) {
//        Scanner sc = new Scanner(System.in);
//        int n = sc.nextInt();
//        int m = sc.nextInt();
//        int k = sc.nextInt();
//        int i = 0,j;
//        char[][] colors = new char[n][m];
//        int[][] nums = new int[n][m];
//        int[][] res = new int[n][m];
//        int temp = n;
//        int temp1, temp2;
//        while (temp>0){
//            String str = sc.next();
//            colors[i] = str.toCharArray();
//            temp--;
//            i++;
//        }
//        for (i = 0; i < n; i++) {
//            for (j = 0; j < m; j++) {
//                nums[i][j] = sc.nextInt();
//            }
//        }
//        res[0][0] = nums[0][0];
//        int maxValue = 0;
//        for (i = 1; i < n; i++) {
//            if(res[i-1][0] < 0){
//                res[i][0] = -1;
//                continue;
//            }
//            if(colors[i-1][0] != colors[i][0]){
//                res[i][0] = res[i-1][0] - k < 0 ? -1 : res[i-1][0] - k + nums[i][0];
//            }else{
//                res[i][0] = res[i-1][0] + nums[i][0];
//            }
//            if(res[i][0]>maxValue){
//                maxValue = res[i][0];
//            }
//        }
//        for (j = 1; j < m; j++) {
//            if(res[0][j-1] < 0){
//                res[0][j] = -1;
//                continue;
//            }
//            if(colors[0][j] != colors[0][j-1]){
//                res[0][j] = res[0][j-1] - k < 0 ? -1 : res[0][j-1] - k + nums[0][j];
//            }else{
//                res[0][j] = res[0][j-1] + nums[0][j];
//            }
//            if(res[0][j]>maxValue){
//                maxValue = res[0][j];
//            }
//        }
//
//        for (i = 1; i < n; i++) {
//            for (j = 1; j < m; j++) {
//                if (res[i - 1][j] < 0 && res[i][j - 1] < 0) {
//                    res[i][j] = -1;
//                    continue;
//                }
//                temp1 = colors[i - 1][j] != colors[i][j] ? res[i - 1][j] - k : res[i - 1][j];
//                temp2 = colors[i][j - 1] != colors[i][j] ? res[i][j - 1] - k : res[i][j - 1];
//                res[i][j] = Math.max(temp1, temp2) < 0 ? -1 : Math.max(temp1, temp2) + nums[i][j];
//                if (res[i][j] > maxValue) {
//                    maxValue = res[i][j];
//                }
//            }
//        }
//        System.out.println(maxValue);
//    }
//    public static void main(String[] args) {
//        Scanner sc = new Scanner(System.in);
//        int n = sc.nextInt();
//        int[] s = new int[n];
//        int[] t = new int[n];
//        int maxTime = 0;
//        int minTime = Integer.MAX_VALUE;
//        for (int i = 0; i < n; i++) {
//            s[i] = sc.nextInt();
//            if(minTime > s[i]){
//                minTime = s[i];
//            }
//        }
//        for (int i = 0; i < n; i++) {
//            t[i] = sc.nextInt();
//            if(maxTime < t[i]){
//                maxTime = t[i];
//            }
//        }
//        int[] nums = new int[maxTime - minTime + 1];
//        int maxCount = Integer.MIN_VALUE;
//        int timeCount = 0;
//        for (int i = 0; i < n; i++) {
//            for (int j = s[i]; j <= t[i]; j++) {
//                nums[j-minTime] += 1;
//                if(nums[j-s[i]] > maxCount){
//                    maxCount = nums[j-s[i]];
//                    timeCount = 1;
//                }else if(nums[j-s[i]] == maxCount){
//                    timeCount += 1;
//                }
//            }
//        }
//        System.out.printf("%d %d\n",maxCount,timeCount);
//    }
}
