package com.lsl.one;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class Solution {
    public int[] twoSum(int[] nums, int target) {
        int[] result = new int[2];
// 双重 for 循环
//        for (int i = 0; i < nums.length-1; i++){
//            for (int j = i+1; j < nums.length; j++) {
//                if ((nums[i] + nums[j]) == target) {
//                    result[0] = i;
//                    result[1] = j;
//                    return result;
//                }
//            }
//        }

// Hash查找
        Map<Integer, Integer> map = new HashMap<>();
        for(int i = 0; i < nums.length;i++){
            map.put(nums[i], i);
        }
        for(int i = 0; i < nums.length; i++){
            int temp = target - nums[i];
            if(map.containsKey(temp) && map.get(temp) != i){
                result[0] = i;
                result[1] = map.get(temp);
                break;
            }
        }
        return result;
    }

    public ListNode addTwoNumbers(ListNode l1, ListNode l2) {
        ListNode head = new ListNode(-1);
        ListNode result = head;
        int sum = 0;
        Boolean carry = false;
        while(l1!=null || l2!=null){
            sum = 0;
            if(l1!=null){
                sum += l1.val;
                l1 = l1.next;
            }
            if(l2!=null){
                sum += l2.val;
                l2 = l2.next;
            }
            if(carry){
                sum ++;
            }
            result.next = new ListNode(sum%10);
            result = result.next;
            carry = sum >= 10;
        }
        if(carry){
            result.next = new ListNode(1);
        }
        return head.next;
    }

    public int lengthOfLongestSubstring(String s) {
        if (s.length()==0) {
            return 0;
        }
        HashMap<Character,Integer> map = new HashMap<Character,Integer>();
        int max = 0;
        int left = 0;
        for (int i = 0; i < s.length(); i++) {
            if(map.containsKey(s.charAt(i))) {
                left = Math.max(left,map.get(s.charAt(i))+1);
            }
            map.put(s.charAt(i), i);
            max = Math.max(max,i-left+1);
        }
        return max;
    }

    public double findMedianSortedArrays(int[] nums1, int[] nums2) {
        int m = nums1.length;
        int n = nums2.length;
        int[] array = new int[m+n];
        int i=0,j=0,index=0;
        while (i < m && j < n){
            if(nums1[i] < nums2[j]){
                array[index] = nums1[i];
                i++;
                index++;
            }else {
                array[index] = nums2[j];
                j++;
                index++;
            }
        }
        if( i < m ){
            for(;i<m;i++){
                array[index] = nums1[i];
                index++;
            }
        }
        if( j < n ){
            for(;j<n;j++){
                array[index] = nums2[j];
                index++;
            }
        }
        if((m+n) % 2 == 0){
            return (array[(m+n)/2] + array[(m+n)/2 -1]) / 2.0;
        }else{
            return array[(m+n)/2];
        }
    }

    public String longestPalindrome(String s) {
        //动态规划
        int len = s.length();
        if(len<2){
            return s;
        }

        int maxLen = 1;
        int begin = 0;
        boolean[][] dp = new boolean[len][len];
        for(int i = 0; i < len; i++){
            dp[i][i] = true;
        }
        char[] charArray = s.toCharArray();
        for (int l=2;l<=len;l++){
            for(int i=0;i<len;i++){
                int j = l + i -1;
                if(j>=len){
                    break;
                }

                if(charArray[i]!=charArray[j]){
                    dp[i][j] = false;
                }else{
                    if(j - i < 3){
                        dp[i][j]=true;
                    }else{
                        dp[i][j]=dp[i+1][j-1];
                    }
                }

                if(dp[i][j] && j - i + 1>maxLen ){
                    maxLen = j - i + 1;
                    begin=i;
                }
            }
        }
        return s.substring(begin,begin+maxLen);
    }

    public String convert(String s, int numRows) {
        if(numRows == 1){
            return s;
        }
        List<StringBuffer> rows = new ArrayList<>();
        for (int i=0;i<Math.min(numRows,s.length());i++){
            rows.add(new StringBuffer());
        }
        int curRow = 0;
        boolean goingDown = false;
        for (char c : s.toCharArray()) {
            rows.get(curRow).append(c);
            if(curRow==0||curRow==numRows-1){
                goingDown = !goingDown;
            }
            curRow += goingDown?1:-1;
        }
        StringBuilder result = new StringBuilder();
        for (StringBuffer row : rows) {
            result.append(row);
        }
        return result.toString();
    }

    public int reverse(int x){
        int rev = 0;
        while (x!=0){
            if(rev<Integer.MIN_VALUE / 10 || rev > Integer.MAX_VALUE/10){
                return 0;
            }
            int digit = x % 10;
            x /= 10;
            rev = rev * 10 + digit;
        }
        return rev;
    }

    public int myAtoi(String s){
        Automaton automaton = new Automaton();
        int length = s.length();
        for (int i=0;i<length;i++){
            automaton.get(s.charAt(i));
        }
        return (int)(automaton.sign * automaton.ans);
    }

    public boolean isPalindrome(int x) {
        int rev = 0;
        if(x<0 || (x%10 == 0  && x != 0)){
            return false;
        }
        while (x > rev ){
            rev = rev * 10 + x % 10;
            x /= 10;
        }
        return rev == x || x == rev / 10;
    }

    public boolean isMatch(String s, String p) {
        int m = s.length();
        int n = p.length();
        boolean[][] f = new boolean[m+1][n+1];
        f[0][0] = true;
        for(int i=0;i<=m;i++){
            for(int j=1;j<=n;j++){
                if(p.charAt(j-1)=='*'){
                    f[i][j] = f[i][j-2];
                    if(matches(s,p,i,j-1)){
                        f[i][j] = f[i][j] || f[i-1][j];
                    }
                }else{
                    if(matches(s,p,i,j)){
                        f[i][j]=f[i-1][j-1];
                    }
                }
            }
        }
        return f[m][n];
    }

    public boolean matches(String s,String p,int i,int j){
        if(i==0){
            return false;
        }
        if(p.charAt(j-1)=='.'){
            return true;
        }
        return s.charAt(i-1) == p.charAt(j-1);
    }
}
