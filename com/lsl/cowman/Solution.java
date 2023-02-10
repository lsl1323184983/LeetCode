package com.lsl.cowman;

import java.util.*;

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
        for (int i = 0; i < nums.length; i++) {
            map.put(nums[i], i);
        }
        for (int i = 0; i < nums.length; i++) {
            int temp = target - nums[i];
            if (map.containsKey(temp) && map.get(temp) != i) {
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
        while (l1 != null || l2 != null) {
            sum = 0;
            if (l1 != null) {
                sum += l1.val;
                l1 = l1.next;
            }
            if (l2 != null) {
                sum += l2.val;
                l2 = l2.next;
            }
            if (carry) {
                sum++;
            }
            result.next = new ListNode(sum % 10);
            result = result.next;
            carry = sum >= 10;
        }
        if (carry) {
            result.next = new ListNode(1);
        }
        return head.next;
    }

    public int lengthOfLongestSubstring(String s) {
        if (s.length() == 0) {
            return 0;
        }
        HashMap<Character, Integer> map = new HashMap<Character, Integer>();
        int max = 0;
        int left = 0;
        for (int i = 0; i < s.length(); i++) {
            if (map.containsKey(s.charAt(i))) {
                left = Math.max(left, map.get(s.charAt(i)) + 1);
            }
            map.put(s.charAt(i), i);
            max = Math.max(max, i - left + 1);
        }
        return max;
    }

    public double findMedianSortedArrays(int[] nums1, int[] nums2) {
        int m = nums1.length;
        int n = nums2.length;
        int[] array = new int[m + n];
        int i = 0, j = 0, index = 0;
        while (i < m && j < n) {
            if (nums1[i] < nums2[j]) {
                array[index] = nums1[i];
                i++;
                index++;
            } else {
                array[index] = nums2[j];
                j++;
                index++;
            }
        }
        if (i < m) {
            for (; i < m; i++) {
                array[index] = nums1[i];
                index++;
            }
        }
        if (j < n) {
            for (; j < n; j++) {
                array[index] = nums2[j];
                index++;
            }
        }
        if ((m + n) % 2 == 0) {
            return (array[(m + n) / 2] + array[(m + n) / 2 - 1]) / 2.0;
        } else {
            return array[(m + n) / 2];
        }
    }

    public String longestPalindrome(String s) {
        //动态规划
        int len = s.length();
        if (len < 2) {
            return s;
        }

        int maxLen = 1;
        int begin = 0;
        boolean[][] dp = new boolean[len][len];
        for (int i = 0; i < len; i++) {
            dp[i][i] = true;
        }
        char[] charArray = s.toCharArray();
        for (int l = 2; l <= len; l++) {
            for (int i = 0; i < len; i++) {
                int j = l + i - 1;
                if (j >= len) {
                    break;
                }

                if (charArray[i] != charArray[j]) {
                    dp[i][j] = false;
                } else {
                    if (j - i < 3) {
                        dp[i][j] = true;
                    } else {
                        dp[i][j] = dp[i + 1][j - 1];
                    }
                }

                if (dp[i][j] && j - i + 1 > maxLen) {
                    maxLen = j - i + 1;
                    begin = i;
                }
            }
        }
        return s.substring(begin, begin + maxLen);
    }

    public String convert(String s, int numRows) {
        if (numRows == 1) {
            return s;
        }
        List<StringBuffer> rows = new ArrayList<>();
        for (int i = 0; i < Math.min(numRows, s.length()); i++) {
            rows.add(new StringBuffer());
        }
        int curRow = 0;
        boolean goingDown = false;
        for (char c : s.toCharArray()) {
            rows.get(curRow).append(c);
            if (curRow == 0 || curRow == numRows - 1) {
                goingDown = !goingDown;
            }
            curRow += goingDown ? 1 : -1;
        }
        StringBuilder result = new StringBuilder();
        for (StringBuffer row : rows) {
            result.append(row);
        }
        return result.toString();
    }

    public int reverse(int x) {
        int rev = 0;
        while (x != 0) {
            if (rev < Integer.MIN_VALUE / 10 || rev > Integer.MAX_VALUE / 10) {
                return 0;
            }
            int digit = x % 10;
            x /= 10;
            rev = rev * 10 + digit;
        }
        return rev;
    }

    public int myAtoi(String s) {
        Automaton automaton = new Automaton();
        int length = s.length();
        for (int i = 0; i < length; i++) {
            automaton.get(s.charAt(i));
        }
        return (int) (automaton.sign * automaton.ans);
    }

    public boolean isPalindrome(int x) {
        int rev = 0;
        if (x < 0 || (x % 10 == 0 && x != 0)) {
            return false;
        }
        while (x > rev) {
            rev = rev * 10 + x % 10;
            x /= 10;
        }
        return rev == x || x == rev / 10;
    }

    public boolean isMatch(String s, String p) {
        int m = s.length();
        int n = p.length();
        boolean[][] f = new boolean[m + 1][n + 1];
        f[0][0] = true;
        for (int i = 0; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                if (p.charAt(j - 1) == '*') {
                    f[i][j] = f[i][j - 2];
                    if (matches(s, p, i, j - 1)) {
                        f[i][j] = f[i][j] || f[i - 1][j];
                    }
                } else {
                    if (matches(s, p, i, j)) {
                        f[i][j] = f[i - 1][j - 1];
                    }
                }
            }
        }
        return f[m][n];
    }

    public boolean matches(String s, String p, int i, int j) {
        if (i == 0) {
            return false;
        }
        if (p.charAt(j - 1) == '.') {
            return true;
        }
        return s.charAt(i - 1) == p.charAt(j - 1);
    }

    public int maxArea(int[] height) {
        int l = 0, r = height.length - 1;
        int res = 0;
        while (l < r) {
            int area = Math.min(height[l], height[r]) * (r - l);
            res = Math.max(res, area);
            if (height[l] <= height[r]) {
                ++l;
            } else {
                --r;
            }
        }
        return res;
    }

//    public String intToRoman(int num) {
//        int digit = 0;
//        Map<Integer, String> map = new HashMap<>();
//        map.put(1,"I");
//        map.put(2,"I");
//        map.put(3,"I");
//        map.put(4,"I");
//        String symbol = "I";
//        int length = 0;
//        while(num >0){
//            length++;
//            digit = num % 10;
//            if(digit)
//        }
//    }

    public String intToRoman(int num) {
        int[] romans = new int[]{1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1};
        String[] romanStr = new String[]{"M", "CM", "D", "CD", "C", "XC", "L", "XL", "X", "IX", "V", "IV", "I"};
        StringBuffer res = new StringBuffer();
        for (int i = 0; i < romans.length; i++) {
            int n = num / romans[i];
            num = num % romans[i];
            for (int j = 0; j < n; j++) {
                res.append(romanStr[i]);
            }
        }
        return res.toString();
    }

    public int romanToInt(String s) {
        int sum = 0;
        for (int i = 0; i < s.length(); ) {
            if (s.charAt(i) == 'M') {
                sum += 1000;
                i++;
            } else if (s.charAt(i) == 'C' && (i + 1) < s.length() && s.charAt(i + 1) == 'M') {
                sum += 900;
                i += 2;
            } else if (s.charAt(i) == 'D') {
                sum += 500;
                i++;
            } else if (s.charAt(i) == 'C' && (i + 1) < s.length() && s.charAt(i + 1) == 'D') {
                sum += 400;
                i += 2;
            } else if (s.charAt(i) == 'C') {
                sum += 100;
                i++;
            } else if (s.charAt(i) == 'X' && (i + 1) < s.length() && s.charAt(i + 1) == 'C') {
                sum += 90;
                i += 2;
            } else if (s.charAt(i) == 'L') {
                sum += 50;
                i++;
            } else if (s.charAt(i) == 'X' && (i + 1) < s.length() && s.charAt(i + 1) == 'L') {
                sum += 40;
                i += 2;
            } else if (s.charAt(i) == 'X') {
                sum += 10;
                i++;
            } else if (s.charAt(i) == 'I' && (i + 1) < s.length() && s.charAt(i + 1) == 'X') {
                sum += 9;
                i += 2;
            } else if (s.charAt(i) == 'V') {
                sum += 5;
                i++;
            } else if (s.charAt(i) == 'I' && (i + 1) < s.length() && s.charAt(i + 1) == 'V') {
                sum += 4;
                i += 2;
            } else {
                sum += 1;
                i++;
            }
        }
        return sum;
    }

    public String longestCommonPrefix(String[] strs) {
        if (strs == null || strs.length == 0) {
            return "";
        }
        String prefix = strs[0];
        int count = strs.length;
        for (int i = 1; i < count; i++) {
            prefix = longestCommonPrefix(prefix, strs[i]);
            if (prefix.length() == 0) {
                return "";
            }
        }
        return prefix;
    }

    public String longestCommonPrefix(String strs1, String strs2) {
        int length = Math.min(strs1.length(), strs2.length());
        int index = 0;
        while (index < length && strs1.charAt(index) == strs2.charAt(index)) {
            index++;
        }
        return strs1.substring(0, index);
    }

    public List<List<Integer>> threeSum(int[] nums) {
        int n = nums.length;
        Arrays.sort(nums);

        ArrayList<List<Integer>> ans = new ArrayList<>();
        for (int first = 0; first < n; first++) {
            if (first > 0 && nums[first] == nums[first - 1]) {
                continue;
            }
            int third = n - 1;
            int taget = -nums[first];
            for (int second = first + 1; second < n; second++) {
                if (second > first + 1 && nums[second] == nums[second - 1]) {
                    continue;
                }
                while (second < third && nums[second] + nums[third] > taget) {
                    --third;
                }
                if (second == third) {
                    break;
                }
                if (nums[second] + nums[third] == taget) {
                    ArrayList<Integer> list = new ArrayList<>();
                    list.add(nums[first]);
                    list.add(nums[second]);
                    list.add(nums[third]);
                    ans.add(list);
                }
            }
        }
        return ans;
    }

    public int threeSumClosest(int[] nums, int target) {
        int best = 10000000;
        Arrays.sort(nums);
        int n = nums.length;
        for (int i = 0; i < n; i++) {
            if (i > 0 && nums[i] == nums[i - 1]) {
                continue;
            }
            int j = i + 1, k = n - 1;
            while (j < k) {
                int sum = nums[i] + nums[j] + nums[k];
                if (sum == target) {
                    return target;
                }

                if (Math.abs(sum - target) < Math.abs(best - target)) {
                    best = sum;
                }

                if (sum > target) {
                    int k0 = k - 1;
                    while (j < k0 && nums[k0] == nums[k]) {
                        --k0;
                    }
                    k = k0;
                } else {
                    int j0 = j + 1;
                    while (j0 < k && nums[j0] == nums[j]) {
                        ++j0;
                    }
                    j = j0;
                }
            }
        }
        return best;
    }

    public List<String> letterCombinations(String digits) {
        List<String> combinations = new ArrayList<>();
        if (digits.length() == 0) {
            return combinations;
        }
        HashMap<Character, String> map = new HashMap<>();
        map.put('2', "abc");
        map.put('3', "def");
        map.put('4', "ghi");
        map.put('5', "jkl");
        map.put('6', "mno");
        map.put('7', "pqrs");
        map.put('8', "tuv");
        map.put('9', "wxyz");
        backtrack(combinations, map, digits, 0, new StringBuffer());
        return combinations;
    }

    public void backtrack(List<String> combinations, HashMap<Character, String> map, String digits, int index, StringBuffer combination) {
        if (index == digits.length()) {
            combinations.add(combination.toString());
        } else {
            char digit = digits.charAt(index);
            String letters = map.get(digit);
            int lettersCount = letters.length();
            for (int i = 0; i < lettersCount; i++) {
                combination.append(letters.charAt(i));
                backtrack(combinations, map, digits, index + 1, combination);
                combination.deleteCharAt(index);
            }
        }
    }

    public List<List<Integer>> fourSum(int[] nums, int target) {
        List<List<Integer>> res = new ArrayList<List<Integer>>();
        if (nums == null || nums.length < 4) {
            return res;
        }
        Arrays.sort(nums);
        int length = nums.length;
        for (int i = 0; i < length - 3; i++) {
            if (i > 0 && nums[i] == nums[i - 1]) {
                continue;
            }
            if ((long) nums[i] + nums[i + 1] + nums[i + 2] + nums[i + 3] > target) {
                break;
            }
            if ((long) nums[i] + nums[length - 1] + nums[length - 2] + nums[length - 3] < target) {
                continue;
            }
            for (int j = i + 1; j < length - 2; j++) {
                if (j > i + 1 && nums[j] == nums[j - 1]) {
                    continue;
                }
                if ((long) nums[i] + nums[j] + nums[j + 1] + nums[j + 2] > target) {
                    break;
                }
                if ((long) nums[i] + nums[j] + nums[length - 1] + nums[length - 2] < target) {
                    continue;
                }
                int left = j + 1, right = length - 1;
                while (left < right) {
                    long sum =(long) nums[i] + nums[j] + nums[left] + nums[right];
                    if (sum == target) {
                        res.add(Arrays.asList(nums[i], nums[j], nums[left], nums[right]));
                        while (left < right && nums[left] == nums[left + 1]) {
                            left++;
                        }
                        left++;
                        while (left < right && nums[right] == nums[right - 1]) {
                            right--;
                        }
                        right--;
                    } else if (sum < target) {
                        left++;
                    } else {
                        right--;
                    }
                }
            }
        }
        return res;
    }

    /**
     * BM1 反转链表
     * @param head
     * @return
     */
    public ListNode ReverseList(ListNode head) {
        ListNode pre = null;
        ListNode cur = head;
        while (cur!=null){
            ListNode cur_next = cur.next;
            cur.next = pre;
            pre = cur;
            cur = cur_next;
        }
        return pre;
    }

    /**
     * BM2 链表内指定区间反转
     * @param head
     * @param m
     * @param n
     * @return
     */
    public ListNode reverseBetween (ListNode head, int m, int n) {
        //设置虚拟头节点
        ListNode res = new ListNode(-1);
        res.next = head;
        ListNode pre = res;
        ListNode cur = head;
        for (int i = 1; i < m; i++) {
            pre = cur;
            cur = cur.next;
        }
        for (int i = m; i < n; i++) {
            ListNode temp = cur.next;
            cur.next = temp.next;
            temp.next = pre.next;
            pre.next = temp;
        }
        return res.next;
    }

    /**
     * BM3 链表中的节点每k个一组翻转
     * @param head
     * @param k
     * @return
     */
    public ListNode reverseKGroup (ListNode head, int k) {
        ListNode tail = head;
        for (int i = 0; i < k; i++) {
            if(tail==null){
                return head;
            }
            tail=tail.next;
        }
        ListNode pre = null;
        ListNode cur = head;
        while (cur!=tail){
            ListNode temp = cur.next;
            cur.next = pre;
            pre = cur;
            cur = temp;
        }
        head.next = reverseKGroup(tail,k);
        return pre;
    }

    /**
     * BM4 合并两个排序的链表
     * @param list1
     * @param list2
     * @return
     */
    public ListNode Merge(ListNode list1,ListNode list2) {
        if(list1==null||list2==null){
            return list1!=null?list1:list2;
        }
        if(list1.val > list2.val){
            list2.next = Merge(list1,list2.next);
            return list2;
        }else{
            list1.next = Merge(list1.next,list2);
            return list1;
        }
    }

    /**
     * BM5 合并k个已排序的链表
     * @param lists
     * @return
     */
    public ListNode mergeKLists(ArrayList<ListNode> lists) {
        return divideMerge(lists,0,lists.size()-1);
    }

    public ListNode divideMerge(ArrayList<ListNode> lists,int left,int right){
        if(left > right){
            return null;
        }
        if(left == right){
            return lists.get(left);
        }
        int mid = (left+right)/2;
        return Merge(divideMerge(lists,left,mid),divideMerge(lists,mid+1,right));
    }

    /**
     * BM6 判断链表中是否有环
     * @param head
     * @return
     */
    public boolean hasCycle(ListNode head) {
        if(head == null){
            return false;
        }
        ListNode fast = head;
        ListNode slow = head;
        while (fast!=null&&fast.next!=null){
            fast = fast.next.next;
            slow = slow.next;
            if(fast == slow){
                return true;
            }
        }
        return false;
    }

    /**
     * BM7 链表中环的入口结点
     * @param pHead
     * @return
     */
    public ListNode EntryNodeOfLoop(ListNode pHead) {
        ListNode slow = hasCycle2(pHead);
        if(slow == null){
            return null;
        }
        ListNode fast = pHead;
        while (fast!=slow){
            fast = fast.next;
            slow = slow.next;
        }
        return slow;
    }

    public ListNode hasCycle2(ListNode head) {
        if(head == null){
            return null;
        }
        ListNode fast = head;
        ListNode slow = head;
        while (fast!=null&&fast.next!=null){
            fast = fast.next.next;
            slow = slow.next;
            if(fast == slow){
                return slow;
            }
        }
        return null;
    }

    /**
     * BM8 链表中倒数最后k个结点
     * @param pHead
     * @param k
     * @return
     */
    public ListNode FindKthToTail (ListNode pHead, int k) {
        ListNode fast = pHead;
        ListNode slow = pHead;
        for (int i = 0; i < k; i++) {
            if(fast!=null){
                fast = fast.next;
            }else {
                return slow = null;
            }
        }
        while (fast!=null){
            fast = fast.next;
            slow = slow.next;
        }
        return slow;
    }

    /**
     * BM9 删除链表的倒数第n个节点
     * @param head
     * @param n
     * @return
     */
    public ListNode removeNthFromEnd (ListNode head, int n) {
        ListNode res = new ListNode(-1);
        res.next = head;
        ListNode cur = head;
        ListNode pre = res;
        ListNode fast = head;
        while (n!=0){
            if(fast!=null){
                fast = fast.next;
                n--;
            }else{
                return res.next;
            }
        }
        while (fast != null){
            fast = fast.next;
            pre = cur;
            cur = cur.next;
        }
        pre.next = cur.next;
        return res.next;
    }

    /**
     * BM10 两个链表的第一个公共结点
     * @param pHead1
     * @param pHead2
     * @return
     */
    public ListNode FindFirstCommonNode(ListNode pHead1, ListNode pHead2) {
        ListNode l1 = pHead1,l2 = pHead2;
        while (l1!=l2){
            l1 = (l1 != null) ? l1.next:pHead2;
            l2 = (l2 != null) ? l2.next:pHead1;
        }
        return l1;
    }

    /**
     * BM11 链表相加(二)
     * @param head1
     * @param head2
     * @return
     */
    public ListNode addInList (ListNode head1, ListNode head2) {
        if(head1 == null){
            return head2;
        }
        if(head2 == null){
            return head1;
        }
        head1 = ReverseList(head1);
        head2 = ReverseList(head2);

        ListNode res = new ListNode(-1);
        ListNode head = res;
        int carry = 0;
        while (head1 != null || head2 != null || carry != 0){
            int val1 = head1 == null ? 0:head1.val;
            int val2 = head2 == null ? 0:head2.val;
            int temp = val1 + val2 + carry;
            carry = temp / 10;
            temp %= 10;
            head.next = new ListNode(temp);
            head = head.next;
            if(head1 != null){
                head1 = head.next;
            }
            if(head2 != null){
                head2 = head.next;
            }
        }
        return ReverseList(res.next);
    }

    /**
     * BM12 单链表的排序
     * @param head
     * @return
     */
    public ListNode sortInList (ListNode head) {
        if(head == null || head.next == null){
            return head;
        }
        ListNode left = head;
        ListNode mid = head.next;
        ListNode right = head.next.next;
        while (right != null && right.next != null){
            left = left.next;
            mid = mid.next;
            right = right.next.next;
        }
        left.next = null;
        return Merge(sortInList(head),sortInList(mid));
    }

    /**
     * BM13 判断一个链表是否为回文结构
     * @param head
     * @return
     */
    public boolean isPail (ListNode head) {
        if(head == null || head.next == null){
            return true;
        }
        ListNode slow = head;
        ListNode fast = head;
        while (fast != null && fast.next != null){
            slow =  slow.next;
            fast = fast.next.next;
        }
        slow = ReverseList(slow);
        fast = head;
        while (slow != null){
            if(slow.val != fast.val){
                return false;
            }
            fast = fast.next;
            slow = slow.next;
        }
        return true;
    }

    /**
     * BM14 链表的奇偶重排
     * @param head
     * @return
     */
    public ListNode oddEvenList (ListNode head) {
        if(head==null){
            return head;
        }
        ListNode even = head.next;
        ListNode odd = head;
        ListNode evenhead = even;
        while (even!=null&&even.next!=null){
            odd.next = even.next;
            odd = odd.next;
            even.next = odd.next;
            even = even.next;
        }
        odd.next = evenhead;
        return head;
    }

    /**
     * BM15 删除有序链表中重复的元素-I
     * @param head
     * @return
     */
    public ListNode deleteDuplicates (ListNode head) {
        if(head==null){
            return null;
        }
        ListNode cur = head;
        while (cur.next != null){
            if(cur.val == cur.next.val){
                cur.next =cur.next.next;
            }else{
                cur = cur.next;
            }
        }
        return head;
    }

    /**
     * BM16 删除有序链表中重复的元素-II
     * @param head
     * @return
     */
    public ListNode deleteDuplicates_2 (ListNode head) {
        if(head == null){
            return null;
        }
        ListNode res = new ListNode(-1);
        res.next = head;
        ListNode cur = res;
        while (cur.next != null && cur.next.next != null){
            if(cur.next.val == cur.next.next.val){
                int temp = cur.next.val;
                while (cur.next != null && cur.next.val == temp){
                    cur.next = cur.next.next;
                }
            }else{
                cur = cur.next;
            }
        }
        return res.next;
    }

    /**
     * BM17 二分查找-I
     * @param nums
     * @param target
     * @return
     */
    public int search (int[] nums, int target) {
        int left = 0;
        int right = nums.length-1;
        int mid = -1;
        while (left <= right){
            mid = (left+right)/2;
            if(nums[mid] == target){
                return mid;
            }
            if(nums[mid] > target){
                right = mid -1;
            }else {
                left = mid + 1;
            }
        }
        return -1;
    }

    /**
     * BM18 二维数组中的查找
     * @param target
     * @param array
     * @return
     */
    public boolean Find(int target, int [][] array) {
        if(array.length == 0){
            return false;
        }
        int n = array.length;
        if(array[0].length==0){
            return false;
        }
        int m = array[0].length;
        for (int i = n-1,j=0; i >= 0 && j < m ;) {
            if(array[i][j] > target) {
                i--;
            } else if (array[i][j] < target) {
                j++;
            }else {
                return true;
            }
        }
        return false;
    }

    /**
     * BM19 寻找峰值
     * @param nums
     * @return
     */
    public int findPeakElement (int[] nums) {
        int left = 0;
        int right = nums.length - 1;
        while (left < right){
            int mid = (left + right) / 2;
            if(nums[mid] > nums[mid+1]){
                right = mid;
            }else {
                left = mid + 1;
            }
        }
        return right;
    }

    /**
     * BM20 数组中的逆序对
     * @param array
     * @return
     */
    int count = 0;
    public int InversePairs(int [] array) {
        if(array.length < 2){
            return 0;
        }
        mergeSort(array,0, array.length-1);
        return count;
    }

    public void mergeSort(int[] array,int left,int right){
        int mid = left + (right - left) / 2;
        if(left < right){
            mergeSort(array,left,mid);
            mergeSort(array,mid+1,right);
            merge(array,left,mid,right);
        }
    }

    public void merge(int[] array,int left,int mid,int right){
        int[] arr = new int[right - left +1];
        int c = 0;
        int s = left;
        int l = left;
        int r = mid +1;
        while (l<=mid && r<=right){
            if(array[l]<=array[r]){
                arr[c] = array[l];
                c++;
                l++;
            }else{
                arr[c] = array[r];
                count += mid+1-l;
                count %= 1000000007;
                c++;
                r++;
            }
        }
        while (l<=mid){
            arr[c++] = array[l++];
        }
        while (r<=right){
            arr[c++] = array[r++];
        }
        for(int num:arr){
            array[s++] = num;
        }
    }

    /**
     * BM21 旋转数组的最小数字
     * @param array
     * @return
     */
    public int minNumberInRotateArray(int [] array) {
        int left = 0;
        int right = array.length - 1;
        while (left < right){
            int mid = (left + right) / 2;
            if(array[mid] > array[right]){
                left = mid + 1;
            }else if(array[mid] == array[right]){
                right--;
            }else {
                right = mid;
            }
        }
        return array[left];
    }

    /**
     * BM22 比较版本号
     * @param version1
     * @param version2
     * @return
     */
    public int compare (String version1, String version2) {
        int n1 = version1.length();
        int n2 = version2.length();
        int i = 0,j = 0;
        while (i < n1 || j < n2 ){
            int num1 = 0;
            while (i < n1 && version1.charAt(i) != '.'){
                num1 = num1 * 10 + (version1.charAt(i) - '0');
                i++;
            }
            i++;
            int num2 = 0;
            while (j < n2 && version2.charAt(j) != '.'){
                num2 = num2 * 10 + (version2.charAt(j) - '0');
                j++;
            }
            j++;
            if(num1 < num2){
                return -1;
            }
            if(num1 > num2){
                return 1;
            }
        }
        return 0;
    }

    /**
     * BM23 二叉树的前序遍历
     * @param root
     * @return
     */
    public int[] preorderTraversal (TreeNode root) {
        List<Integer> list = new ArrayList<>();
        preorder(list,root);
        int[] res = new int[list.size()];
        for (int i = 0; i < list.size(); i++) {
            res[i] = list.get(i);
        }
        return res;
    }

    public void preorder(List<Integer> list, TreeNode root){
        if(root == null){
            return;
        }
        list.add(root.val);
        preorder(list,root.left);
        preorder(list,root.right);
    }

    /**
     * BM24 二叉树的中序遍历
     * @param root
     * @return
     */
    public int[] inorderTraversal (TreeNode root) {
        List<Integer> list = new ArrayList<>();
        inorder(list,root);
        int[] res = new int[list.size()];
        for (int i = 0; i < list.size(); i++) {
            res[i] = list.get(i);
        }
        return res;
    }

    public void inorder(List<Integer> list, TreeNode root){
        if(root == null){
            return;
        }
        inorder(list,root.left);
        list.add(root.val);
        inorder(list,root.right);
    }

    /**
     * BM25 二叉树的后序遍历
     * @param root
     * @return
     */
    public int[] postorderTraversal (TreeNode root) {
        List<Integer> list = new ArrayList<>();
        postorder(list,root);
        int[] res = new int[list.size()];
        for (int i = 0; i < list.size(); i++) {
            res[i] = list.get(i);
        }
        return res;
    }

    public void postorder(List<Integer> list,TreeNode root){
        if(root == null){
            return;
        }
        postorder(list,root.left);
        postorder(list,root.right);
        list.add(root.val);
    }


}
