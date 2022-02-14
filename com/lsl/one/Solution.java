package com.lsl.one;

import java.util.HashMap;
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
}
